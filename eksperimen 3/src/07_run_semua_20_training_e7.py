from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
CFG = SRC / '07_training_config_e7_LOCKED.json'
EXPECTED_CFG_SHA256 = 'cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'
MODELS = (('frozen', 'raw'), ('frozen', 'clahe'), ('lora', 'raw'), ('lora', 'clahe'))


def check_config() -> None:
    if not CFG.is_file():
        raise RuntimeError(f'Konfigurasi terkunci tidak ditemukan: {CFG}')
    actual = hashlib.sha256(CFG.read_bytes()).hexdigest()
    if actual != EXPECTED_CFG_SHA256:
        raise RuntimeError('SHA256 konfigurasi E7 tidak cocok. Hentikan dan periksa paket training.')
    cfg = json.loads(CFG.read_text(encoding='utf-8'))
    if cfg.get('status') != 'LOCKED_E7_FOUR_MODELS':
        raise RuntimeError('Status konfigurasi E7 tidak sesuai paket empat model.')
    for method in ('frozen', 'lora'):
        if not (SRC / f'07_train_{method}_5fold_e7.py').is_file():
            raise RuntimeError(f'Skrip training {method} tidak ditemukan di {SRC}')
    if not (SRC / '07_status_training_e7.py').is_file():
        raise RuntimeError('07_status_training_e7.py tidak ditemukan.')


def status_for_run(method: str, variant: str, fold: int) -> str:
    folder = ROOT / 'output' / 'training' / f'{method}_{variant}_fold_{fold}'
    summary = folder / 'run_summary.json'
    latest = folder / 'latest_training_state.pt'
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding='utf-8'))
            expected = {
                'status': 'PASS_E7_FOLD_TRAINING_COMPLETE',
                'experiment': 'experiment7',
                'model': f'{method}_{variant}',
                'fold': fold,
                'locked_config_sha256': EXPECTED_CFG_SHA256,
                'test_read': False,
            }
            for k, v in expected.items():
                if data.get(k) != v:
                    raise ValueError(f'field {k} tidak cocok')
            score = data.get('best_validation_auroc')
            if not (isinstance(data.get('best_epoch'), int) and data['best_epoch'] > 0 and
                    isinstance(score, (int, float)) and 0 <= score <= 1):
                raise ValueError('best epoch / AUROC tidak valid')
            required = ('training_history.csv', 'best_validation_predictions.csv')
            if method == 'frozen':
                required += ('best_classifier.pt',)
                base = folder
            else:
                pointer = data.get('best_snapshot_relative_path')
                if not isinstance(pointer, str) or not pointer:
                    raise ValueError('best_snapshot_relative_path tidak ada')
                relative = Path(pointer)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('path snapshot LoRA tidak aman')
                base = folder / relative
                if not all((base / name).is_file() for name in
                           ('adapter_model.safetensors', 'adapter_config.json',
                            'best_classifier.pt', 'best_validation_predictions.csv')):
                    raise ValueError('snapshot LoRA terbaik belum lengkap')
            # history and predictions are in the run folder for Frozen; for
            # LoRA best validation predictions are stored in the snapshot.
            must_exist = ('training_history.csv',) if method == 'lora' else required
            if not all((folder / name).is_file() for name in must_exist):
                raise ValueError('file hasil training tidak lengkap')
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f'Run {method}_{variant} Fold {fold} memiliki '
                               f'run_summary.json tidak valid: {exc}') from exc
        return 'SKIP (COMPLETE)'
    if latest.is_file():
        return 'RESUME'
    if folder.exists():
        return 'BLOCKED (folder ada, checkpoint tidak tersedia)'
    return 'START'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Periksa rencana tanpa training')
    parser.add_argument('--only', choices=('all', 'frozen', 'lora'), default='all',
                        help='Jalankan hanya Frozen atau LoRA jika diperlukan')
    args = parser.parse_args()

    check_config()
    models = [m for m in MODELS if args.only == 'all' or m[0] == args.only]
    plans = []
    print(f'EXPERIMENT 7 | Python: {sys.executable}', flush=True)
    print(f'Folder skrip: {SRC}', flush=True)
    for method, variant in models:
        for fold in range(1, 6):
            operation = status_for_run(method, variant, fold)
            print(f'{method:6} {variant:5} Fold {fold}: {operation}', flush=True)
            plans.append((method, variant, fold, operation))
    if args.check:
        print('CHECK SELESAI. Tidak ada training dijalankan.', flush=True)
        return 0 if not any(p[3].startswith('BLOCKED') for p in plans) else 2

    for method, variant, fold, _ in plans:
        operation = status_for_run(method, variant, fold)
        if operation.startswith('SKIP'):
            print(f'[SKIP] {method} {variant} Fold {fold}', flush=True)
            continue
        if operation.startswith('BLOCKED'):
            print(f'[STOP] {method} {variant} Fold {fold}: folder sudah ada tanpa '
                  'checkpoint yang dapat dipulihkan. Jangan hapus folder.', file=sys.stderr)
            return 2
        script = SRC / f'07_train_{method}_5fold_e7.py'
        cmd = [sys.executable, str(script), '--fold', str(fold), '--variant', variant]
        if operation == 'RESUME':
            cmd.append('--resume')
        print(f'[{operation}] {method} {variant} Fold {fold}', flush=True)
        try:
            result = subprocess.run(cmd, cwd=SRC, check=False)
        except KeyboardInterrupt:
            print('\nDihentikan. Untuk lanjut, jalankan perintah Python yang sama.',
                  file=sys.stderr, flush=True)
            return 130
        if result.returncode != 0:
            print(f'[STOP] {method} {variant} Fold {fold} gagal '
                  f'(exit code {result.returncode}).', file=sys.stderr, flush=True)
            return result.returncode
        if not status_for_run(method, variant, fold).startswith('SKIP'):
            print(f'[STOP] Training {method} {variant} Fold {fold} selesai tanpa '
                  'run_summary.json yang valid.', file=sys.stderr, flush=True)
            return 3

    print('STATUS TRAINING E7:', flush=True)
    # For --only, do not require all 20 folds complete.
    cmd = [sys.executable, str(SRC / '07_status_training_e7.py')]
    if args.only == 'all':
        cmd.append('--require-complete')
    return subprocess.run(cmd, cwd=SRC, check=False).returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f'FAIL_E7_RUNNER: {exc}', file=sys.stderr)
        sys.exit(2)
