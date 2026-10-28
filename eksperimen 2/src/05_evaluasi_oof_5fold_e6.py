from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
FOLD_ROOT = ROOT / 'data/manifests/experiment6_85dev_15test_5fold_seed42_v1'
TRAIN_ROOT = ROOT / 'output/training'
OUT = ROOT / 'output/cross_validation/05_oof_model_selection_v1'
CONFIG = SRC / '04_training_config_e6_LOCKED.json'
AUDIT = ROOT / 'audits/03_dataloader_validation/03_e6_dataloader_validation_report.json'
LOCK = FOLD_ROOT / '02_e6_5fold_lock_report.json'
DEV = FOLD_ROOT / 'e6_development_1887_5fold_LOCKED.csv'
KEY = ['source_archive', 'archive_member_path']
IDENTITY = KEY + ['patient_id', 'y_true', 'cv_fold']
MODELS = [('frozen','raw'), ('frozen','clahe'), ('lora','raw'), ('lora','clahe')]
STATUS = 'PASS_E6_OOF_MODEL_SELECTION_LOCKED'
CONFIG_SHA = '30c609b8762722da0fc196584d8dfcf9fa3efd3d7dbe07afacdc791e25f4d564'
SELECTION_FILE = '05_e6_oof_model_selection_LOCKED.json'
OUTPUT_FILES = ['05_e6_fold_metrics_20runs.csv','05_e6_oof_comparison_4models.csv'] + [
    f'05_e6_oof_predictions_{a}_{b}.csv' for a,b in MODELS
]


def ensure(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jread(path):
    ensure(Path(path).is_file(), f'Berkas JSON tidak ada: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def cread(path):
    ensure(Path(path).is_file(), f'Berkas CSV tidak ada: {path}')
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def equal_numeric(actual, expected, context, tol=1e-9):
    ensure(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
           math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol),
           f'Metrik tidak cocok ({context}): {actual} vs {expected}')


def image_set(df):
    return set(map(tuple, df[KEY].to_numpy()))


def measurements(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ensure(len(y) == len(p) and len(y) > 0 and
           np.isfinite(p).all() and np.logical_and(p >= 0, p <= 1).all() and
           set(y.tolist()) == {0, 1}, 'Probabilitas/label tidak valid atau hanya satu kelas')
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, pred, labels=[0,1]).ravel()]
    return {
        'auroc':float(roc_auc_score(y,p)),
        'accuracy':float(np.mean(pred == y)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'sensitivity_dr':float(tp/(tp+fn)),
        'specificity_ds':float(tn/(tn+fp)),
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'threshold':.5,
    }


def sources():
    ensure(digest(CONFIG) == CONFIG_SHA, 'Konfigurasi training E6 telah berubah')
    config = jread(CONFIG)
    ensure(config['status'] == 'LOCKED_PROTOCOL_E6_FROM_E5_2026-10-22' and
           config['folds'] == 5 and config['cross_validation'] is True and
           config['primary_model_selection_after_cv'] == 'out_of_fold_AUROC' and
           config['final_inference'] == 'five_checkpoint_probability_ensemble',
           'Protokol E6 berbeda dari yang terkunci')
    lock = jread(LOCK)
    audit = jread(AUDIT)
    ensure(lock['status'] == 'PASS_E6_FOLDS_LOCKED' and
           audit['status'] == 'PASS_E6_DATALOADER_VALIDATION' and
           audit['n_combinations_checked'] == 20, 'Audit fold/DataLoader belum lulus')
    ensure(digest(LOCK) == config['locked_e6_fold_report_sha256'] ==
           audit['source_fold_report_sha256'], 'Checksum laporan penguncian fold berbeda')
    ensure(digest(DEV) == config['locked_e6_fold_manifest_sha256'],
           'Manifest development telah berubah')
    ensure(lock['test_manifest_sha256'] == config['e5_test_manifest_sha256'] ==
           audit['e5_fixed_holdout_test_sha256'],
           'Referensi checksum test E5 berbeda; TEST TIDAK DIBACA')
    ensure(lock['no_patient_leakage'] is True and
           lock['oof_each_development_image_exactly_once'] is True and
           lock['oof_each_development_patient_exactly_once'] is True,
           'Audit OOF awal tidak memenuhi syarat')
    dev = cread(DEV)
    ensure(set(IDENTITY).issubset(dev.columns) and 'e5_original_split' in dev.columns,
           'Kolom development E6 tidak lengkap')
    ensure(len(dev)==1887 and dev.patient_id.nunique()==1471 and
           dev['y_true'].value_counts().to_dict()=={'0':944,'1':943} and
           dev.e5_original_split.isin(['train','validation']).all() and
           dev.cv_fold.isin(['1','2','3','4','5']).all() and
           not dev[KEY].duplicated().any() and
           dev.groupby('patient_id').cv_fold.nunique().eq(1).all() and
           dev.groupby('patient_id').y_true.nunique().eq(1).all(),
           'Dataset development berubah, ada duplikasi/leakage/label salah')
    for fold in range(1,6):
        rel = f'fold_{fold}/validation.csv'
        path = FOLD_ROOT / rel
        ensure(digest(path) == lock['output_sha256'][rel],f'Hash manifest {rel} berbeda')
        val = cread(path)
        ensure(set(IDENTITY + ['split']).issubset(val.columns),
               f'Kolom manifest fold {fold} tidak lengkap')
        expect = dev.loc[dev.cv_fold==str(fold)]
        ensure(val.split.eq('validation').all() and
               len(expect)==len(val) and image_set(expect)==image_set(val) and
               not val[KEY].duplicated().any(), f'Partisi validation fold {fold} tidak sesuai')
        ident = expect[IDENTITY].merge(val[IDENTITY], on=KEY, suffixes=('_dev','_val'),
                                       validate='one_to_one')
        for col in ('patient_id','y_true','cv_fold'):
            ensure(ident[f'{col}_dev'].eq(ident[f'{col}_val']).all(),
                   f'Identitas {col} manifest validation fold {fold} tidak cocok')
    return config, lock, audit, dev, {
        'training_config':digest(CONFIG),
        'fold_lock_report':digest(LOCK),
        'fold_development_manifest':digest(DEV),
        'dataloader_audit':digest(AUDIT),
        'fixed_test_manifest_reference_SHA256_NO_TEST_READ':config['e5_test_manifest_sha256'],
    }


def inspect_one(method, variant, fold, config, lock, audit, dev, source_hashes):
    name=f'{method}_{variant}'
    out=TRAIN_ROOT / f'{name}_fold_{fold}'
    files=['run_summary.json','run_config.json','training_history.csv',
           'best_validation_metrics.json','best_validation_predictions.csv',
           'best_classifier.pt']
    if method=='lora':
        files+=['best_adapter/adapter_model.safetensors',
                'best_adapter/adapter_config.json']
    for filename in files:
        f=out/filename
        ensure(f.is_file() and f.stat().st_size>0,f'Run belum lengkap: {name} F{fold} -> {f}')
    s=jread(out/'run_summary.json')
    run=jread(out/'run_config.json')
    reported=jread(out/'best_validation_metrics.json')
    hist=cread(out/'training_history.csv')
    pred=cread(out/'best_validation_predictions.csv')
    ensure(s.get('status')=='PASS_E6_FOLD_TRAINING_COMPLETE' and
           s.get('experiment')=='experiment6' and
           s.get('method')==method and s.get('variant')==variant and
           s.get('fold')==fold and
           s.get('five_fold_cv') is True and
           s.get('test_pixel_read') is False and
           s.get('test_inference') is False,
           f'Run summary tidak valid: {name} F{fold}')
    ensure(run.get('method')==method and run.get('variant')==variant and
           run.get('fold')==fold and run.get('test_data_used') is False,
           f'Run config tidak cocok: {name} F{fold}')
    ensure(run.get('seed')==42 and run.get('max_epochs')==30 and
           run.get('patience')==5 and float(run.get('min_delta',-1))==.001 and
           float(run.get('lr_head',-1))==.001 and
           float(run.get('weight_decay',-1))==.0001,
           f'Hyperparameter umum berubah: {name} F{fold}')
    if method=='frozen':
        ensure(run.get('head_batch_size')==64,f'Head batch Frozen berubah {name} F{fold}')
    else:
        ensure(run.get('physical_batch')==1 and run.get('gradient_accumulation')==16 and
               run.get('effective_batch')==16 and run.get('gradient_checkpointing') is False and
               float(run.get('lr_lora',-1))==.0001 and
               run.get('lora_layers_0_based')==[8,9,10,11] and
               run.get('lora_projections')==['query','value'] and
               run.get('lora_r')==8 and run.get('lora_alpha')==16 and
               float(run.get('lora_dropout',-1))==.05,
               f'Konfigurasi LoRA berubah: {name} F{fold}')
    expected_fp=run.get('training_fingerprints',{})
    ensure(s.get('fingerprints')==expected_fp and
           expected_fp.get('training_config')==source_hashes['training_config'] and
           expected_fp.get('stage2_e6_folds')==source_hashes['fold_lock_report'] and
           expected_fp.get('stage3_e6_dataloader')==source_hashes['dataloader_audit'] and
           expected_fp.get('e6_dev_locked')==source_hashes['fold_development_manifest'] and
           expected_fp.get('e5_test_manifest')==config['e5_test_manifest_sha256'] and
           expected_fp.get('e5_processed_manifest')==config['expected_hashes']['e5_processed_manifest_sha256'] and
           expected_fp.get('fold')==fold,
           f'Sumber/data provenance run tidak sesuai: {name} F{fold}')
    rel=f'fold_{fold}/validation.csv'
    ensure(expected_fp.get('fold_validation_manifest')==lock['output_sha256'][rel] and
           expected_fp.get('fold_train_manifest')==lock['output_sha256'][f'fold_{fold}/train.csv'],
           f'Manifest split run tidak sesuai: {name} F{fold}')
    expected = dev.loc[dev.cv_fold==str(fold)]
    ensure(s.get('validation_images')==len(expected) and
           expected_fp.get('validation_images')==len(expected) and
           s.get('train_images')==1887-len(expected) and
           expected_fp.get('train_images')==1887-len(expected),
           f'Jumlah gambar fold tidak sesuai: {name} F{fold}')
    ensure(s.get('frozen_head_batch_size')==64 and
           s.get('lora_micro_batch_size')==1 and
           s.get('lora_gradient_accumulation')==16,
           f'Konfigurasi batch summary berubah: {name} F{fold}')
    ensure(s.get('model_id')=='microsoft/rad-dino' and
           s.get('model_commit')=='110cbc18d5133582e320b43d53bf5c44e410c936' and
           run.get('model_commit')==s.get('model_commit'),
           f'Revisi RAD-DINO tidak cocok: {name} F{fold}')
    ensure(set(IDENTITY+['p_dr','pred_0_5','split']).issubset(pred.columns),
           f'Kolom prediksi tidak lengkap: {name} F{fold}')
    ensure(len(pred)==len(expected) and pred.split.eq('validation').all() and
           pred.cv_fold.eq(str(fold)).all() and
           not pred[KEY].duplicated().any() and image_set(pred)==image_set(expected),
           f'OOF predictions tidak sesuai dengan fold: {name} F{fold}')
    both=expected[IDENTITY].merge(pred[IDENTITY],on=KEY,
                                  suffixes=('_expected','_pred'),validate='one_to_one')
    for c in ('patient_id','y_true','cv_fold'):
        ensure(both[f'{c}_expected'].eq(both[f'{c}_pred']).all(),
               f'Identitas prediksi {c} salah: {name} F{fold}')
    ensure(pred.y_true.isin(['0','1']).all() and pred.pred_0_5.isin(['0','1']).all(),
           f'Label/prediction invalid: {name} F{fold}')
    probabilities = pd.to_numeric(pred.p_dr, errors='coerce').to_numpy()
    ensure(np.isfinite(probabilities).all() and (probabilities>=0).all() and
           (probabilities<=1).all() and
           np.array_equal((probabilities>=.5).astype(int),pred.pred_0_5.astype(int).to_numpy()),
           f'Nilai p_dr atau threshold 0.5 salah: {name} F{fold}')
    met=measurements(pred.y_true.astype(int), probabilities)
    ensure(int(reported['best_epoch'])==int(s['best_epoch']) and
           1<=int(s['best_epoch'])<=int(s['last_epoch'])<=30 and
           int(s['epochs_recorded'])==int(s['last_epoch'])==len(hist),
           f'Best epoch/history tidak cocok: {name} F{fold}')
    ensure(hist.epoch.tolist()==[str(x) for x in range(1,len(hist)+1)],
           f'Riwayat epoch bukan urutan kontinu: {name} F{fold}')
    best_hist=hist.iloc[int(s['best_epoch'])-1]
    equal_numeric(best_hist['val_auroc'],met['auroc'],f'history val AUROC {name} F{fold}')
    equal_numeric(s['best_validation_auroc'],met['auroc'],f'run AUROC {name} F{fold}')
    for mk,mval in met.items():
        if mk=='threshold':
            equal_numeric(reported[mk],mval,f'report threshold {name} F{fold}')
        elif mk in ('tn','fp','fn','tp'):
            ensure(int(reported[mk])==mval,f'Confusion matrix berubah {name} F{fold}/{mk}')
        else:
            equal_numeric(reported[mk],mval,f'metrics {name} F{fold}/{mk}')
    hashes={x:digest(out/x) for x in files}
    row={'model':name,'method':method,'variant':variant,'fold':fold,
         'n_validation_images':len(pred),
         'n_validation_patients':int(pred.patient_id.nunique()),
         'n_ds':int(pred.y_true.eq('0').sum()),'n_dr':int(pred.y_true.eq('1').sum()),
         'best_epoch':int(s['best_epoch']),'last_epoch':int(s['last_epoch']),**met}
    pred=pred[IDENTITY+['p_dr','pred_0_5']].copy()
    pred['p_dr']=probabilities
    pred['model']=name
    return row, pred, hashes


def compute_all():
    config, lock, audit, dev, source_hashes=sources()
    fold_rows=[]; model_rows=[]; predictions={}; provenance={}
    for method,variant in MODELS:
        name=f'{method}_{variant}'
        all_predictions=[]; folded=[]
        for fold in range(1,6):
            row, pred, hashes=inspect_one(method,variant,fold,config,lock,audit,dev,source_hashes)
            fold_rows.append(row)
            folded.append(row)
            all_predictions.append(pred)
            provenance[f'{name}_fold_{fold}']=hashes
        oof=pd.concat(all_predictions,ignore_index=True)
        ensure(len(oof)==1887 and oof.patient_id.nunique()==1471 and
               not oof[KEY].duplicated().any() and image_set(oof)==image_set(dev),
               f'OOF harus mencakup tepat 1887 gambar: {name}')
        pairs=dev[IDENTITY].merge(oof[IDENTITY],on=KEY,
                                  suffixes=('_locked','_oof'),validate='one_to_one')
        for col in ('patient_id','y_true','cv_fold'):
            ensure(pairs[f'{col}_locked'].eq(pairs[f'{col}_oof']).all(),
                   f'OOF identitas {col} tidak sesuai pada {name}')
        ensure(oof.y_true.value_counts().to_dict()=={'0':944,'1':943},
               f'Distribusi kelas OOF tidak cocok: {name}')
        oof=oof.sort_values(KEY).reset_index(drop=True)
        metrics=measurements(oof.y_true.astype(int).to_numpy(),oof.p_dr.to_numpy())
        fold_auc=np.array([r['auroc'] for r in folded])
        model_rows.append({'model':name,'method':method,'variant':variant,
                           'oof_images':1887,'oof_patients':1471,
                           'n_folds':5,'fold_auroc_mean':float(fold_auc.mean()),
                           'fold_auroc_sd_sample':float(fold_auc.std(ddof=1)),
                           'fold_auroc_min':float(fold_auc.min()),
                           'fold_auroc_max':float(fold_auc.max()),
                           'oof_auroc':metrics['auroc'],
                           'oof_macro_f1':metrics['macro_f1'],
                           'oof_balanced_accuracy':metrics['balanced_accuracy'],
                           'oof_sensitivity_dr':metrics['sensitivity_dr'],
                           'oof_specificity_ds':metrics['specificity_ds'],
                           'oof_accuracy':metrics['accuracy'],
                           'tn':metrics['tn'],'fp':metrics['fp'],
                           'fn':metrics['fn'],'tp':metrics['tp'],
                           'threshold':.5})
        predictions[name]=oof
    fold_df=pd.DataFrame(fold_rows)
    cmp_df=pd.DataFrame(model_rows).sort_values(
        ['oof_auroc','fold_auroc_mean','model'],
        ascending=[False,False,True],kind='stable').reset_index(drop=True)
    primary=cmp_df.iloc[0]['model']
    report={'status':STATUS,'experiment':'experiment6',
            'protocol':'fixed E5 15% test; 5-fold CV on 85% development',
            'method':'aggregate saved best-fold validation probabilities WITHOUT new inference',
            'selection_criterion':'maximum pooled image-level OOF AUROC',
            'tie_breakers':['higher mean fold AUROC','alphabetic model name'],
            'primary_model':primary,
            'primary_model_auroc':float(cmp_df.iloc[0]['oof_auroc']),
            'primary_selected_before_e6_testing':True,
            'n_development_images':1887,'n_development_patients':1471,
            'n_test_images_REFERENCE_ONLY':333,'n_models':4,'n_training_runs':20,
            'source_sha256':source_hashes,'run_artifact_sha256':provenance,
            'models':cmp_df.to_dict('records'),
            'test_pixel_read':False,'test_predictions_read':False,
            'test_inference_performed':False,'new_training_performed':False,
            'notes':[
                'OOF metrics are at IMAGE level; patient-level aggregation is not performed.',
                'Each fold checkpoint was selected on that SAME validation fold; OOF estimates can be optimistic. This is not nested CV.',
                'Fold AUROC SD is descriptive (sample SD, ddof=1), not a confidence interval.',
                'The E5 15% test set was already evaluated in E5 and is a reused exploratory comparison holdout, not a new never-seen test.',
                'Five-checkpoint probability ensemble on fixed E5 test belongs in Stage 6, not here.',
            ]}
    return fold_df,cmp_df,predictions,report


def write_all(fold_df,cmp_df,preds,report):
    ensure(not OUT.exists(),f'Folder tahap 5 sudah ada, tidak menimpa: {OUT}. Jalankan --verify.')
    OUT.mkdir(parents=True,exist_ok=False)
    fold_df.to_csv(OUT/OUTPUT_FILES[0],index=False)
    cmp_df.to_csv(OUT/OUTPUT_FILES[1],index=False)
    for (method,variant) in MODELS:
        name=f'{method}_{variant}'
        preds[name].to_csv(OUT/f'05_e6_oof_predictions_{name}.csv',index=False)
    report['output_sha256']={filename:digest(OUT/filename) for filename in OUTPUT_FILES}
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    (OUT/SELECTION_FILE).write_text(text,encoding='utf-8')
    (OUT/(SELECTION_FILE+'.sha256')).write_text(digest(OUT/SELECTION_FILE)+'  '+SELECTION_FILE+'\n',encoding='utf-8')
    return report


def verify_all():
    report=jread(OUT/SELECTION_FILE)
    ensure(report.get('status')==STATUS,'Hasil pemilihan OOF tidak terkunci')
    seal=(OUT/(SELECTION_FILE+'.sha256')).read_text(encoding='utf-8').strip()
    ensure(seal==digest(OUT/SELECTION_FILE)+'  '+SELECTION_FILE,
           'SHA256 segel pemilihan berbeda')
    for fname in OUTPUT_FILES:
        ensure(digest(OUT/fname)==report['output_sha256'][fname],
               f'Berkas hasil OOF berubah: {fname}')
    _, cmp_df, _, current=compute_all()
    ensure(current['source_sha256']==report['source_sha256'] and
           current['run_artifact_sha256']==report['run_artifact_sha256'] and
           current['models']==report['models'] and
           current['primary_model']==report['primary_model'],
           'Sumber atau pemilihan OOF sudah berubah setelah dikunci')
    ensure(cmp_df.iloc[0]['model']==report['primary_model'],
           'Model primer bukan pemenang OOF AUROC')
    print(f'PASS_E6_OOF_MODEL_SELECTION_VERIFIED | selected={report["primary_model"]} | test not used')


def show(cmp_df):
    showcols=['model','fold_auroc_mean','fold_auroc_sd_sample','oof_auroc',
              'oof_macro_f1','oof_balanced_accuracy','oof_sensitivity_dr','oof_specificity_ds']
    print('\nE6 5-FOLD EVALUATION: 1887 DEVELOPMENT IMAGES | HOLDOUT TEST NOT READ')
    print(cmp_df[showcols].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nPRIMARY CONFIGURATION (OOF AUROC):',cmp_df.iloc[0]['model'])
    print('NOTE: fold checkpoints selected on same fold-validation; OOF could be optimistic.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true',help='Audit and preview only; no output writing')
    group.add_argument('--run',action='store_true',help='Audit all 20 then write immutable OOF selection')
    group.add_argument('--verify',action='store_true',help='Re-audit sources and verify saved output hashes')
    args=ap.parse_args()
    if args.verify:
        verify_all()
        return
    fold_df,cmp_df,preds,report=compute_all()
    show(cmp_df)
    if args.check:
        print('\nPASS_E6_OOF_PREFLIGHT_20_RUNS | no outputs written')
        return
    write_all(fold_df,cmp_df,preds,report)
    print('\n'+STATUS+' | output:',OUT)
    verify_all()


if __name__=='__main__':
    try:
        main()
    except Exception as err:
        print(f'E6 STAGE 5 FAILED: {type(err).__name__}: {err}',file=sys.stderr,flush=True)
        raise SystemExit(1)
