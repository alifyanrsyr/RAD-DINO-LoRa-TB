from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / 'output' / 'cross_validation' / '08_e7_oof_model_selection_v1'
OUT = ROOT / 'output' / 'threshold' / '11_e7_threshold_oof_review_v1'
LOCK_NAME = '08_e7_oof_model_selection_LOCKED.json'
COMP_NAME = '08_e7_oof_comparison_4models.csv'
REPORT_NAME = '11_e7_threshold_report.json'
STATUS = 'PASS_E7_EXPLORATORY_OOF_THRESHOLD_REVIEW'
MODS = ['frozen_raw', 'frozen_clahe', 'lora_raw', 'lora_clahe']
KEY = ['source_archive', 'archive_member_path']
CORE = ['patient_id', 'y_true', 'cv_fold', 'p_dr', 'pred_0_5']
# Parameter analisis deskriptif; bukan parameter training atau threshold baru.
GRID = [round(n / 100, 2) for n in range(20, 81)]  # 0.20 .. 0.80
BASELINE = 0.5


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jread(path: Path) -> dict:
    require(path.is_file(), f'File tidak ditemukan: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def close(a, b, label: str) -> None:
    require(math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-8),
            f'Nilai {label} tidak sesuai lock: {a} != {b}')


def scores(y, probability, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=float)
    require(len(y) == len(probability) and len(y) > 0 and
            set(np.unique(y).tolist()) == {0, 1},
            'Label/ukuran input metrik tidak valid')
    pred = (probability >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        'threshold': float(threshold),
        'auroc': float(roc_auc_score(y, probability)),
        'accuracy': float(np.mean(pred == y)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'sensitivity_dr': float(tp / (tp + fn)),
        'specificity_ds': float(tn / (tn + fp)),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def load_verified_oof():
    lock = jread(INP / LOCK_NAME)
    require(lock.get('status') == 'PASS_E7_OOF_MODEL_SELECTION_LOCKED' and
            lock.get('primary_model') == 'lora_clahe' and
            lock.get('primary_selected_before_e7_test') is True and
            lock.get('n_development_images') == 1887 and
            lock.get('n_development_patients') == 1471 and
            lock.get('n_test_images_REFERENCE_ONLY') == 333 and
            lock.get('n_training_runs') == 20 and
            lock.get('test_pixel_read') is False and
            lock.get('test_predictions_read') is False and
            lock.get('test_inference_performed') is False,
            'Lock OOF tidak sesuai protokol E7')

    file_names = [COMP_NAME] + [f'08_e7_oof_predictions_{model}.csv' for model in MODS]
    for name in file_names:
        path = INP / name
        require(path.is_file() and sha256(path) == lock['output_sha256'][name],
                f'Hash locked berubah atau file hilang: {name}')

    comp = pd.read_csv(INP / COMP_NAME)
    require(len(comp) == 4 and set(comp['model']) == set(MODS),
            'Perbandingan OOF tidak memuat tepat empat model')
    require((comp['threshold'].astype(float) == 0.5).all(),
            'Baseline threshold berubah dari 0.5')
    preds = {}
    ref = None
    for model in MODS:
        name = f'08_e7_oof_predictions_{model}.csv'
        df = pd.read_csv(INP / name, dtype={col: str for col in KEY + ['patient_id']}, low_memory=False)
        require(set(KEY + CORE).issubset(df.columns),
                f'Kolom identitas/probabilitas tidak lengkap: {name}')
        require(len(df) == 1887 and df['patient_id'].nunique() == 1471 and
                not df[KEY].isna().any().any() and not df[KEY].duplicated().any(),
                f'Identitas citra/pasien tidak sesuai: {name}')
        for col in ['y_true', 'cv_fold', 'pred_0_5']:
            df[col] = pd.to_numeric(df[col], errors='raise').astype(np.int64)
        df['p_dr'] = pd.to_numeric(df['p_dr'], errors='raise').astype(float)
        require(df['y_true'].value_counts().to_dict() == {0: 944, 1: 943} and
                df['cv_fold'].between(1, 5).all() and
                df['patient_id'].notna().all() and
                df.groupby('patient_id')['cv_fold'].nunique().eq(1).all() and
                df.groupby('patient_id')['y_true'].nunique().eq(1).all() and
                np.isfinite(df['p_dr']).all() and df['p_dr'].between(0, 1).all() and
                np.array_equal(df['pred_0_5'].to_numpy(),
                               (df['p_dr'].to_numpy() >= BASELINE).astype(int)),
                f'Pemeriksaan label, fold, atau probabilitas gagal: {name}')
        for fold in range(1, 6):
            require(set(df.loc[df.cv_fold.eq(fold), 'y_true']) == {0, 1},
                    f'Terdapat fold satu kelas pada {model}, fold {fold}')
        df = df.sort_values(KEY).reset_index(drop=True)
        if ref is None:
            ref = df[KEY + ['patient_id', 'y_true', 'cv_fold']].copy()
        else:
            require(df[KEY + ['patient_id', 'y_true', 'cv_fold']].equals(ref),
                    f'Pasien/label/fold tidak identik dengan model lain: {model}')

        original = comp.loc[comp.model.eq(model)].iloc[0]
        baseline = scores(df.y_true, df.p_dr, BASELINE)
        for k in ['auroc', 'macro_f1', 'balanced_accuracy', 'sensitivity_dr',
                  'specificity_ds', 'accuracy']:
            close(baseline[k], original[f'oof_{k}'], f'{model}: {k}')
        for k in ['tn', 'fp', 'fn', 'tp']:
            require(baseline[k] == int(original[k]), f'{model}: {k} tidak sesuai')
        preds[model] = df
    require(comp.sort_values('oof_auroc', ascending=False).iloc[0]['model'] == 'lora_clahe',
            'Model utama tidak konsisten dengan pooled OOF AUROC')
    return lock, comp, preds


def prefer_near_baseline(rows: pd.DataFrame, criterion: str) -> dict:
    """Candidate only. Tie: closest to fixed 0.5, then lower threshold."""
    copy = rows.copy()
    copy['dist_0_5'] = (copy['threshold'] - BASELINE).abs()
    best = copy.sort_values([criterion, 'dist_0_5', 'threshold'],
                            ascending=[False, True, True], kind='stable').iloc[0]
    return {k: float(best[k]) if k not in ['tn', 'fp', 'fn', 'tp'] else int(best[k])
            for k in ['threshold', criterion, 'sensitivity_dr', 'specificity_ds',
                      'macro_f1', 'balanced_accuracy', 'tn', 'fp', 'fn', 'tp']}


def compute():
    lock, comp, preds = load_verified_oof()
    all_rows = []
    for mod in MODS:
        df = preds[mod]
        for t in GRID:
            all_rows.append({'model': mod, **scores(df.y_true, df.p_dr, t)})
    sweep = pd.DataFrame(all_rows)
    m4 = sweep.loc[sweep.model.eq('lora_clahe')].reset_index(drop=True)
    candidate = prefer_near_baseline(m4, 'balanced_accuracy')

    # Diagnostic threshold stability: choose grid candidate on four OOF folds
    # then inspect held fold. NOT unbiased independent test/nested CV: the
    # checkpoints themselves were selected using validation AUROC.
    df = preds['lora_clahe']
    frows = []
    decisions = np.zeros(len(df), dtype=np.int64)
    base_predictions = (df.p_dr.to_numpy() >= BASELINE).astype(int)
    for held in range(1, 6):
        keep = df.cv_fold.to_numpy() != held
        hold = ~keep
        fit_y, fit_p = df.y_true.to_numpy()[keep], df.p_dr.to_numpy()[keep]
        fit_rows = pd.DataFrame([scores(fit_y, fit_p, t) for t in GRID])
        fit = prefer_near_baseline(fit_rows, 'balanced_accuracy')
        t = fit['threshold']
        hold_metrics = scores(df.y_true.to_numpy()[hold], df.p_dr.to_numpy()[hold], t)
        hold_baseline = scores(df.y_true.to_numpy()[hold], df.p_dr.to_numpy()[hold], BASELINE)
        decisions[hold] = (df.p_dr.to_numpy()[hold] >= t).astype(int)
        frows.append({
            'held_fold': held,
            'n_held_images': int(hold.sum()),
            'n_fit_images': int(keep.sum()),
            'threshold_fit_on_other_4_oof_folds': float(t),
            'fit_balanced_accuracy': float(fit['balanced_accuracy']),
            'held_balanced_accuracy_at_fit_threshold': hold_metrics['balanced_accuracy'],
            'held_sensitivity_dr_at_fit_threshold': hold_metrics['sensitivity_dr'],
            'held_specificity_ds_at_fit_threshold': hold_metrics['specificity_ds'],
            'held_macro_f1_at_fit_threshold': hold_metrics['macro_f1'],
            'held_fn_at_fit_threshold': hold_metrics['fn'],
            'held_fp_at_fit_threshold': hold_metrics['fp'],
            'held_balanced_accuracy_at_baseline': hold_baseline['balanced_accuracy'],
            'held_sensitivity_dr_at_baseline': hold_baseline['sensitivity_dr'],
            'held_specificity_ds_at_baseline': hold_baseline['specificity_ds'],
        })
    fresult = pd.DataFrame(frows)
    y = df.y_true.to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, decisions, labels=[0, 1]).ravel()]
    crossfit = {
        'balanced_accuracy': float(balanced_accuracy_score(y, decisions)),
        'macro_f1': float(f1_score(y, decisions, average='macro', zero_division=0)),
        'sensitivity_dr': float(tp / (tp + fn)),
        'specificity_ds': float(tn / (tn + fp)),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'threshold_per_fold': {str(int(r.held_fold)): float(r.threshold_fit_on_other_4_oof_folds)
                               for _, r in fresult.iterrows()},
    }
    require(len(GRID) == 61 and BASELINE in GRID and
            np.array_equal(base_predictions, df['pred_0_5'].to_numpy()),
            'Baseline prediction atau grid threshold tidak benar')
    return lock, comp, sweep, m4, candidate, fresult, crossfit


def figures(m4, folder: Path) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for field, label in [('sensitivity_dr', 'Sensitivitas DR-TB'),
                         ('specificity_ds', 'Spesifisitas DS-TB'),
                         ('balanced_accuracy', 'Balanced Accuracy'),
                         ('macro_f1', 'Macro-F1')]:
        ax.plot(m4.threshold, m4[field], label=label, linewidth=1.8)
    ax.axvline(BASELINE, color='gray', linestyle='--', label='Baseline 0,50')
    ax.set(xlabel='Threshold probabilitas DR-TB', ylabel='Nilai metrik',
           title='Eksperimen 7 — M4 LoRA CLAHE: Analisis Threshold OOF')
    ax.set_ylim(0, 1)
    ax.grid(alpha=.2); ax.legend(loc='best'); fig.tight_layout()
    n = '11_e7_m4_threshold_tradeoff.png'
    fig.savefig(folder/n, dpi=180); plt.close(fig); files.append(n)

    fig, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(m4.threshold, m4.fn, label='False Negative (DR-TB terlewat)', linewidth=2)
    ax.plot(m4.threshold, m4.fp, label='False Positive (DS-TB salah DR-TB)', linewidth=2)
    ax.axvline(BASELINE, color='gray', linestyle='--', label='Baseline 0,50')
    ax.set(xlabel='Threshold probabilitas DR-TB', ylabel='Jumlah citra',
           title='Eksperimen 7 — M4 LoRA CLAHE: Trade-off Kesalahan OOF')
    ax.grid(alpha=.2); ax.legend(loc='best'); fig.tight_layout()
    n = '11_e7_m4_fn_fp_vs_threshold.png'
    fig.savefig(folder/n, dpi=180); plt.close(fig); files.append(n)
    return files


def summarize(lock, comp, sweep, candidate, foldreview, crossfit):
    row05 = sweep.loc[sweep.model.eq('lora_clahe') & sweep.threshold.eq(BASELINE)].iloc[0]
    return {
        'status': STATUS,
        'experiment': 'experiment7',
        'primary_model_from_locked_oof': lock['primary_model'],
        'primary_model_selection_unchanged': True,
        'baseline_threshold_unchanged': BASELINE,
        'analysis_grid': {'min': GRID[0], 'max': GRID[-1], 'step': 0.01,
                          'includes_baseline': True},
        'patient_grouped_oof': True,
        'n_development_images': 1887,
        'n_development_patients': 1471,
        'n_folds': 5,
        'label_positive': 'DR-TB (1)',
        'baseline_m4': {k: (int(row05[k]) if k in ['tn','fp','fn','tp'] else float(row05[k]))
                        for k in ['threshold','auroc','accuracy','macro_f1',
                                  'balanced_accuracy','sensitivity_dr',
                                  'specificity_ds','tn','fp','fn','tp']},
        'full_oof_balanced_accuracy_best_candidate_EXPLORATORY_NOT_LOCKED': candidate,
        'other_fold_threshold_fit_review_EXPLORATORY_NOT_INDEPENDENT_TEST': crossfit,
        'five_fold_candidate_threshold_range': {
            'minimum': float(foldreview.threshold_fit_on_other_4_oof_folds.min()),
            'maximum': float(foldreview.threshold_fit_on_other_4_oof_folds.max()),
            'median': float(foldreview.threshold_fit_on_other_4_oof_folds.median()),
        },
        'no_threshold_adopted_for_holdout': True,
        'no_test_data_read': True,
        'no_training': True,
        'no_model_inference': True,
        'methodological_limits': [
            'Threshold sweep and full OOF candidate are exploratory, not an unbiased optimized threshold estimate.',
            'Checkpoint was selected using AUROC on its validation fold (OOF selection optimism).',
            'Four-fold OOF threshold fitting and fifth-fold review are diagnostic, not independent nested validation.',
            'Patient grouping prevents within-fold patient overlap, but all reported metrics are image-level.',
            'The threshold search grid does not guarantee a global optimum.',
            'The reused historical E5 holdout is not a completely fresh independent test.',
            'Clinical cost of false negatives versus false positives is not specified; maximizing balanced accuracy is only an exploratory reference.',
        ],
        'source_sha256': {
            'oof_locked_json': sha256(INP / LOCK_NAME),
            'oof_comparison_csv': sha256(INP / COMP_NAME),
            'all_four_oof_csvs': {
                m: sha256(INP / f'08_e7_oof_predictions_{m}.csv') for m in MODS
            }
        },
    }


def check():
    lock, comp, sweep, m4, candidate, foldreview, crossfit = compute()
    r05 = m4.loc[m4.threshold.eq(BASELINE)].iloc[0]
    print('\nE7 | DEVELOPMENT OOF THRESHOLD ANALYSIS | TANPA TEST')
    print(f'Model utama terkunci: {lock["primary_model"]}; threshold baseline: {BASELINE:.2f}')
    print(f'Jumlah development: {len(sweep) // len(GRID) * 1887 // 4} citra per model; '
          '5 fold, 1.471 pasien')
    print('M4 pada threshold 0,50 | AUROC %.6f | Sens DR %.6f | Spes DS %.6f | '
          'Macro-F1 %.6f | Balanced Acc %.6f | FN %d | FP %d' %
          (r05.auroc, r05.sensitivity_dr, r05.specificity_ds,
           r05.macro_f1, r05.balanced_accuracy, int(r05.fn), int(r05.fp)))
    print('Threshold dengan balanced accuracy tertinggi pada grid OOF (EKSPLORATIF): '
          f'{candidate["threshold"]:.2f}; belum disetujui/dikunci untuk test.')
    print('Threshold dari pelatihan pada 4 fold OOF (diagnostik, bukan nested CV):',
          ', '.join('F%d: %.2f' % (int(r.held_fold),r.threshold_fit_on_other_4_oof_folds)
                    for _,r in foldreview.iterrows()))
    print('PREFLIGHT PASS | Tidak ada data testing dibaca, tidak ada training/inferensi.')


def run():
    require(not OUT.exists(),
            f'Folder output sudah ada: {OUT}. Gunakan --verify; tidak ditimpa otomatis.')
    lock, comp, sweep, m4, candidate, foldreview, crossfit = compute()
    OUT.mkdir(parents=True, exist_ok=False)
    saved = []
    for name, data in [
        ('11_e7_threshold_sweep_4models.csv', sweep),
        ('11_e7_m4_threshold_analysis.csv', m4),
        ('11_e7_m4_foldwise_threshold_robustness.csv', foldreview),
    ]:
        data.to_csv(OUT/name, index=False, float_format='%.12g')
        saved.append(name)
    saved.extend(figures(m4, OUT))
    report = summarize(lock, comp, sweep, candidate, foldreview, crossfit)
    report['output_sha256'] = {name: sha256(OUT/name) for name in saved}
    (OUT / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                               allow_nan=False) + '\n', encoding='utf-8')
    (OUT / (REPORT_NAME+'.sha256')).write_text(
        sha256(OUT/REPORT_NAME)+'  '+REPORT_NAME+'\n', encoding='utf-8')
    print(STATUS)
    print('Hasil tersimpan di:', OUT)
    print('Threshold baseline 0,50 tetap dikunci; threshold alternatif BELUM diadopsi.')


def verify():
    report = jread(OUT/REPORT_NAME)
    require(report.get('status') == STATUS and report.get('no_test_data_read') is True and
            report.get('no_threshold_adopted_for_holdout') is True,
            'Status atau invariant laporan threshold tidak benar')
    sig = (OUT/(REPORT_NAME+'.sha256')).read_text(encoding='utf-8').strip()
    require(sig == sha256(OUT/REPORT_NAME)+'  '+REPORT_NAME,
            'Checksum laporan tidak cocok')
    for name, digest in report['output_sha256'].items():
        require((OUT/name).is_file() and sha256(OUT/name) == digest,
                f'Checksum output tidak cocok: {name}')
    lock, comp, sweep, m4, candidate, foldreview, crossfit = compute()
    current = summarize(lock, comp, sweep, candidate, foldreview, crossfit)
    for k,v in current.items():
        require(report.get(k) == v, f'Isi laporan tidak cocok dengan source: {k}')
    print(STATUS, '| VERIFIED | No test data read |', OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Validasi sumber dan preview tanpa menyimpan')
    group.add_argument('--run', action='store_true', help='Buat laporan/grafik sekali, tanpa menimpa')
    group.add_argument('--verify', action='store_true', help='Verifikasi SHA-256 dan hitung ulang metrik')
    args = ap.parse_args()
    if args.check:
        check()
    elif args.run:
        run()
    else:
        verify()


if __name__ == '__main__':
    main()
