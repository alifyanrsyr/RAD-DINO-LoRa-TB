from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
E6_ROOT = ROOT.parent / 'eksperimen6'
FOLDS = E6_ROOT / 'data/manifests/experiment6_85dev_15test_5fold_seed42_v1'
DEV = FOLDS / 'e6_development_1887_5fold_LOCKED.csv'
LOCK = FOLDS / '02_e6_5fold_lock_report.json'
E7_CONFIG = SRC / '07_training_config_e7_LOCKED.json'
TRAINING = ROOT / 'output/training'
OUT = ROOT / 'output/cross_validation/08_e7_oof_model_selection_v1'

LOCKED_CONFIG_SHA256 = 'cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'
MODELS = [('frozen', 'raw'), ('frozen', 'clahe'), ('lora', 'raw'), ('lora', 'clahe')]
KEY = ['source_archive', 'archive_member_path']
IDENTITY = KEY + ['patient_id', 'y_true', 'cv_fold']
METRICS = ['auroc','accuracy','macro_f1','balanced_accuracy', 'sensitivity_dr',
           'specificity_ds','tn','fp','fn','tp','threshold']
FILE_FOLDS = '08_e7_fold_metrics_20runs.csv'
FILE_COMPARE = '08_e7_oof_comparison_4models.csv'
FILE_REPORT = '08_e7_oof_model_selection_LOCKED.json'
FILES_PRED = [f'08_e7_oof_predictions_{m}_{v}.csv' for m,v in MODELS]
RESULT_FILES = [FILE_FOLDS, FILE_COMPARE] + FILES_PRED
STATUS = 'PASS_E7_OOF_MODEL_SELECTION_LOCKED'


def require(ok, why):
    if not bool(ok):
        raise RuntimeError(why)


def sha256(p: Path):
    h = hashlib.sha256()
    with Path(p).open('rb') as fh:
        for chunk in iter(lambda:fh.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def jread(path: Path):
    require(path.is_file(), f'Berkas JSON tidak ditemukan: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def cread(path: Path):
    require(path.is_file(), f'Berkas CSV tidak ditemukan: {path}')
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def eqnum(x, y, what, tol=1e-8):
    require(math.isfinite(float(x)) and math.isfinite(float(y)) and
            math.isclose(float(x), float(y), abs_tol=tol, rel_tol=tol),
            f'Nilai berbeda {what}: {x} vs {y}')


def image_keys(df):
    return set(map(tuple, df[KEY].to_numpy()))


def measure(y, p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    require(len(y)>0 and len(y)==len(p) and np.isfinite(p).all() and
            (p>=0).all() and (p<=1).all() and set(y.tolist())=={0,1},
            'Prediksi/label tidak valid untuk penghitungan metrik')
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=[int(n) for n in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    return {'auroc':float(roc_auc_score(y,p)),
            'accuracy':float(np.mean(y==pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'sensitivity_dr':float(tp/(tp+fn)),
            'specificity_ds':float(tn/(tn+fp)),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp,'threshold':.5}


def source_check():
    require(E7_CONFIG.is_file() and sha256(E7_CONFIG)==LOCKED_CONFIG_SHA256,
            'Konfigurasi E7 berubah/tidak ditemukan')
    config=jread(E7_CONFIG)
    require(config.get('status')=='LOCKED_E7_FOUR_MODELS' and
            config.get('experiment')=='experiment7' and
            config.get('n_splits')==5 and
            config.get('development_images')==1887 and
            config.get('development_patients')==1471 and
            config.get('heldout_test_images_reference_only')==333 and
            config.get('decision_threshold_baseline')==.5 and
            config.get('checkpoint_selection')=='per_fold_validation_AUROC' and
            config.get('test_pixels_are_off_limits') is True,
            'Protokol E7 tidak sesuai LOCKED')
    require([(v['method'],v['variant']) for v in config['model_variants']]==MODELS,
            'Urutan model/varian tidak sesuai E7 LOCKED')
    lock=jread(LOCK)
    require(lock.get('status')=='PASS_E6_FOLDS_LOCKED' and
            lock.get('no_patient_leakage') is True and
            lock.get('oof_each_development_image_exactly_once') is True and
            lock.get('oof_each_development_patient_exactly_once') is True,
            'E6 fold lock tidak valid')
    require(DEV.is_file() and
            sha256(DEV)==config['reference_e6_fold_manifest_sha256'],
            'Manifest development E6 telah berubah')
    require(lock['test_manifest_sha256']==config['reference_e5_test_manifest_sha256'],
            'Referensi hash holdout tidak cocok (test TIDAK dibaca)')
    dev=cread(DEV)
    require(set(IDENTITY).issubset(dev.columns) and
            len(dev)==1887 and dev.patient_id.nunique()==1471 and
            not dev[KEY].duplicated().any() and
            dev.cv_fold.isin(['1','2','3','4','5']).all() and
            dev.y_true.value_counts().to_dict()=={'0':944,'1':943} and
            dev.groupby('patient_id').cv_fold.nunique().eq(1).all() and
            dev.groupby('patient_id').y_true.nunique().eq(1).all() and
            dev.e5_original_split.isin(['train','validation']).all(),
            'Manifest development tidak memenuhi integrity/patient-level split')
    val_folds={}
    for fold in range(1,6):
        rel=f'fold_{fold}/validation.csv'
        path=FOLDS/rel
        require(path.is_file() and sha256(path)==lock['output_sha256'][rel],
                f'Manifest validation fold {fold} berubah')
        va=cread(path)
        exp=dev.loc[dev.cv_fold.eq(str(fold))]
        require(set(IDENTITY+['split']).issubset(va.columns) and
                len(va)==len(exp) and
                va.split.eq('validation').all() and
                not va[KEY].duplicated().any() and
                image_keys(va)==image_keys(exp),
                f'Validation fold {fold} tidak sesuai manifest')
        merged=exp[IDENTITY].merge(va[IDENTITY], on=KEY,
                                  suffixes=('_exp','_val'),validate='one_to_one')
        for col in ('patient_id','y_true','cv_fold'):
            require(merged[f'{col}_exp'].eq(merged[f'{col}_val']).all(),
                    f'Identitas {col} fold {fold} berubah')
        val_folds[fold]=va
    source_sha={
        'e7_training_config':sha256(E7_CONFIG),
        'e6_fold_lock':sha256(LOCK),
        'e6_development_manifest':sha256(DEV),
        'e5_fixed_holdout_manifest_reference_ONLY':config['reference_e5_test_manifest_sha256'],
    }
    return config, lock, dev, val_folds, source_sha


def safe_rel(root: Path, path_text: str):
    rel=Path(path_text)
    require(not rel.is_absolute() and '..' not in rel.parts and
            len(rel.parts)>=1, f'Path snapshot tidak aman: {path_text}')
    return root/rel


def fold_check(method,variant,fold,config,lock,dev,validation):
    model=f'{method}_{variant}'
    root=TRAINING/f'{model}_fold_{fold}'
    paths=['run_summary.json','run_config.json','training_history.csv']
    for name in paths:
        require((root/name).is_file() and (root/name).stat().st_size>0,
                f'File training E7 hilang: {root/name}')
    s=jread(root/'run_summary.json')
    rc=jread(root/'run_config.json')
    require(s.get('status')=='PASS_E7_FOLD_TRAINING_COMPLETE' and
            s.get('experiment')=='experiment7' and
            s.get('model')==model and s.get('fold')==fold and
            s.get('test_read') is False and
            s.get('training_only_development') is True and
            s.get('locked_config_sha256')==LOCKED_CONFIG_SHA256,
            f'Status/integritas training tidak valid: {model} fold {fold}')
    require(rc.get('experiment')=='experiment7' and
            rc.get('fold')==fold and rc.get('test_read') is False,
            f'run_config tidak sesuai {model} fold {fold}')
    if method=='frozen':
        require(s.get('method')==method and s.get('variant')==variant and
                rc.get('model')==model and
                rc.get('config_sha256')==LOCKED_CONFIG_SHA256,
                f'Metadata Frozen E7 tidak sesuai: {model} fold {fold}')
        base=root
        paths+=['best_validation_predictions.csv','best_validation_metrics.json',
                'best_classifier.pt']
    else:
        require(rc.get('method')=='lora' and rc.get('variant')==variant and
                rc.get('locked_config_sha256')==LOCKED_CONFIG_SHA256,
                f'Metadata LoRA E7 tidak sesuai: {model} fold {fold}')
        pointer=jread(root/'best_checkpoint.json')
        require(int(pointer['epoch'])==int(s['best_epoch']) and
                pointer['snapshot_relative_path']==s['best_snapshot_relative_path'],
                f'Pointer checkpoint LoRA tidak sesuai: {model} fold {fold}')
        base=safe_rel(root, pointer['snapshot_relative_path'])
        require(base.parent==root/'best_candidates' and
                base.name==f"epoch_{int(s['best_epoch']):02d}",
                f'Folder LoRA best epoch tidak sesuai: {model} fold {fold}')
        for k, expected in [('predictions_path',base/'best_validation_predictions.csv'),
                            ('metrics_path',base/'best_validation_metrics.json'),
                            ('classifier_path',base/'best_classifier.pt'),
                            ('adapter_path',base)]:
            require(safe_rel(root,pointer[k])==expected,
                    f'Pointer {k} menunjuk file salah: {model} fold {fold}')
        eqnum(pointer['validation_auroc'],s['best_validation_auroc'],
              f'LoRA best pointer {model} F{fold}')
        paths+=['best_checkpoint.json',
                str((base/'best_validation_predictions.csv').relative_to(root)),
                str((base/'best_validation_metrics.json').relative_to(root)),
                str((base/'best_classifier.pt').relative_to(root)),
                str((base/'adapter_model.safetensors').relative_to(root)),
                str((base/'adapter_config.json').relative_to(root))]
    for rel in paths:
        path=root/rel
        require(path.is_file() and path.stat().st_size>0,
                f'Run E7 kurang berkas: {model} Fold {fold}, {rel}')
    fp=s['source_e6_fingerprints']
    require(rc.get('source_e6_fingerprints',rc.get('fold_source_fingerprints'))==fp and
            fp.get('fold')==fold and
            fp.get('e6_dev_locked')==config['reference_e6_fold_manifest_sha256'] and
            fp.get('stage2_e6_folds')==sha256(LOCK) and
            fp.get('e5_test_manifest')==config['reference_e5_test_manifest_sha256'] and
            fp.get('fold_validation_manifest')==lock['output_sha256'][f'fold_{fold}/validation.csv'] and
            fp.get('fold_train_manifest')==lock['output_sha256'][f'fold_{fold}/train.csv'],
            f'Source fingerprint mismatch: {model} Fold {fold}')
    hist=cread(root/'training_history.csv')
    pred=cread(base/'best_validation_predictions.csv')
    reported=jread(base/'best_validation_metrics.json')
    require(set(IDENTITY+['split','p_dr','pred_0_5']).issubset(pred.columns),
            f'Kolom prediksi validation tidak lengkap: {model} Fold {fold}')
    require(not pred[KEY].duplicated().any() and
            pred.split.eq('validation').all() and
            pred.cv_fold.eq(str(fold)).all() and
            len(pred)==len(validation) and
            image_keys(pred)==image_keys(validation),
            f'Prediksi validasi tidak sesuai manifest fold: {model} Fold {fold}')
    pairs=validation[IDENTITY].merge(pred[IDENTITY],on=KEY,
                                     suffixes=('_locked','_pred'),validate='one_to_one')
    for col in ('patient_id','y_true','cv_fold'):
        require(pairs[f'{col}_locked'].eq(pairs[f'{col}_pred']).all(),
                f'{col} tidak sesuai pada prediksi {model} Fold {fold}')
    require(pred.y_true.isin(['0','1']).all(),
            f'Label prediksi tidak valid: {model} Fold {fold}')
    p=pred.p_dr.astype(float).to_numpy()
    require(pred.pred_0_5.astype(int).eq((p>=.5).astype(int)).all(),
            f'Keputusan threshold 0.5 salah: {model} Fold {fold}')
    metrics=measure(pred.y_true.astype(int).to_numpy(),p)
    epoch=int(s['best_epoch'])
    last=int(s['last_epoch'])
    require(1<=epoch<=last<=config['max_epochs'] and
            len(hist)==last==int(s['epochs_recorded']) and
            hist.epoch.tolist()==[str(n) for n in range(1,last+1)],
            f'Riwayat epoch tidak sesuai: {model} Fold {fold}')
    eqnum(hist.iloc[epoch-1]['val_auroc'],metrics['auroc'],f'history AUROC {model} F{fold}')
    eqnum(s['best_validation_auroc'],metrics['auroc'],f'summary AUROC {model} F{fold}')
    require(int(reported['best_epoch'])==epoch,
            f'Best epoch metrik tidak konsisten: {model} Fold {fold}')
    for mk,value in metrics.items():
        if mk in ['tn','fp','fn','tp']:
            require(int(reported[mk])==value,
                    f'{mk} tidak konsisten: {model} Fold {fold}')
        else:
            eqnum(reported[mk],value,f'{mk} {model} F{fold}')
    hashes={rel:sha256(root/rel) for rel in paths}
    row={'model':model,'method':method,'variant':variant,'fold':fold,
         'n_validation_images':len(pred),'n_validation_patients':pred.patient_id.nunique(),
         'n_ds':int(pred.y_true.eq('0').sum()),'n_dr':int(pred.y_true.eq('1').sum()),
         'best_epoch':epoch,'last_epoch':last,**metrics}
    result=pred[IDENTITY+['p_dr','pred_0_5']].copy()
    result['p_dr']=p
    result['model']=model
    return row,result,hashes


def compute_all():
    config,lock,dev,val_folds,source_sha=source_check()
    fold_rows=[]
    model_rows=[]
    outputs={}
    artifact_hashes={}
    for method,variant in MODELS:
        name=f'{method}_{variant}'
        preds=[]; rows=[]
        for fold in range(1,6):
            row,pred,hashes=fold_check(method,variant,fold,config,lock,dev,val_folds[fold])
            fold_rows.append(row)
            rows.append(row)
            preds.append(pred)
            artifact_hashes[f'{name}_fold_{fold}']=hashes
        oof=pd.concat(preds,ignore_index=True)
        require(len(oof)==1887 and oof.patient_id.nunique()==1471 and
                not oof[KEY].duplicated().any() and
                image_keys(oof)==image_keys(dev),
                f'OOF {name} harus berisi tepat 1887 citra development')
        merged=dev[IDENTITY].merge(oof[IDENTITY],on=KEY,
                                   suffixes=('_locked','_oof'),validate='one_to_one')
        for col in ('patient_id','y_true','cv_fold'):
            require(merged[f'{col}_locked'].eq(merged[f'{col}_oof']).all(),
                    f'OOF {col} tidak sesuai: {name}')
        require(oof.y_true.value_counts().to_dict()=={'0':944,'1':943},
                f'OOF label count mismatch: {name}')
        oof=oof.sort_values(KEY).reset_index(drop=True)
        met=measure(oof.y_true.astype(int).to_numpy(),oof.p_dr.to_numpy())
        auc=np.array([r['auroc'] for r in rows])
        model_rows.append({
            'model':name,'method':method,'variant':variant,
            'oof_images':1887,'oof_patients':1471,'n_folds':5,
            'fold_auroc_mean':float(auc.mean()),
            'fold_auroc_sd_sample':float(auc.std(ddof=1)),
            'fold_auroc_min':float(auc.min()),
            'fold_auroc_max':float(auc.max()),
            'oof_auroc':met['auroc'],'oof_macro_f1':met['macro_f1'],
            'oof_balanced_accuracy':met['balanced_accuracy'],
            'oof_sensitivity_dr':met['sensitivity_dr'],
            'oof_specificity_ds':met['specificity_ds'],
            'oof_accuracy':met['accuracy'],
            'tn':met['tn'],'fp':met['fp'],'fn':met['fn'],'tp':met['tp'],
            'threshold':.5,
        })
        outputs[name]=oof
    fold_df=pd.DataFrame(fold_rows)
    cmp_df=pd.DataFrame(model_rows).sort_values(
        ['oof_auroc','fold_auroc_mean','model'],
        ascending=[False,False,True],kind='stable').reset_index(drop=True)
    primary=cmp_df.iloc[0]['model']
    report={
        'status':STATUS,'experiment':'experiment7',
        'protocol':'E6 85% development patient-level 5-fold; E5 15% fixed holdout isolated',
        'method':'Audit immutable per-fold best validation predictions; no new model inference',
        'selection_criterion':'maximum pooled image-level OOF AUROC',
        'tie_breakers':['higher mean fold AUROC','alphabetically smaller model name'],
        'primary_model':primary,
        'primary_model_oof_auroc':float(cmp_df.iloc[0]['oof_auroc']),
        'primary_selected_before_e7_test':True,
        'n_development_images':1887,'n_development_patients':1471,
        'n_test_images_REFERENCE_ONLY':333,
        'n_models':4,'n_training_runs':20,
        'source_sha256':source_sha,
        'run_artifact_sha256':artifact_hashes,
        'models':cmp_df.to_dict(orient='records'),
        'test_pixel_read':False,'test_predictions_read':False,
        'test_inference_performed':False,'new_training_performed':False,
        'notes':[
            'All evaluations are image-level. Patients are grouped by validation fold, without patient overlap.',
            'Best checkpoints were selected on their validation fold; resulting OOF metrics may have checkpoint-selection optimism (not nested CV).',
            'Fold AUROC SD is descriptive sample SD (ddof=1), not a confidence interval.',
            'E5 fixed holdout previously used in earlier experiments; subsequent testing is exploratory reuse.',
            'Do not use test labels to choose model or optimize threshold.',
        ],
    }
    return fold_df,cmp_df,outputs,report


def show(cmp_df):
    cols=['model','fold_auroc_mean','fold_auroc_sd_sample','oof_auroc',
          'oof_macro_f1','oof_balanced_accuracy','oof_sensitivity_dr','oof_specificity_ds']
    print('\nE7 5-FOLD OOF | 1887 DEVELOPMENT IMAGES | NO TEST INPUT')
    print(cmp_df[cols].to_string(index=False,float_format=lambda f:f'{f:.6f}'))
    print('\nPRIMARY SELECTED ON POOLED OOF AUROC:',cmp_df.iloc[0]['model'])
    print('NOTE: Selection on fold validation AUROC may bias OOF performance upward.')


def write_all(folds,compare,preds,report):
    require(not OUT.exists(),f'Output OOF sudah ada: {OUT}. Jangan menimpa; gunakan --verify.')
    OUT.mkdir(parents=True,exist_ok=False)
    folds.to_csv(OUT/FILE_FOLDS,index=False)
    compare.to_csv(OUT/FILE_COMPARE,index=False)
    for method,variant in MODELS:
        name=f'{method}_{variant}'
        preds[name].to_csv(OUT/f'08_e7_oof_predictions_{name}.csv',index=False)
    report['output_sha256']={name:sha256(OUT/name) for name in RESULT_FILES}
    (OUT/FILE_REPORT).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/(FILE_REPORT+'.sha256')).write_text(sha256(OUT/FILE_REPORT)+'  '+FILE_REPORT+'\n',encoding='utf-8')
    print('\nPASS_E7_OOF_MODEL_SELECTION_LOCKED | outputs:',OUT)


def verify():
    report=jread(OUT/FILE_REPORT)
    require(report.get('status')==STATUS,'Status lock hasil E7 OOF tidak valid')
    sig=(OUT/(FILE_REPORT+'.sha256')).read_text(encoding='utf-8').strip()
    require(sig==sha256(OUT/FILE_REPORT)+'  '+FILE_REPORT,
            'Segel SHA256 model selection berubah')
    for name in RESULT_FILES:
        require(sha256(OUT/name)==report['output_sha256'][name],
                f'Hasil OOF berubah: {name}')
    _,cmp_df,_,now=compute_all()
    for key in ('source_sha256','run_artifact_sha256','models','primary_model'):
        require(now[key]==report[key],f'Sumber dan hasil pilihan tidak cocok: {key}')
    print('PASS_E7_OOF_MODEL_SELECTION_VERIFIED | selected=',report['primary_model'], '| test not used')
    show(cmp_df)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Audit 20 runs, hitung OOF, preview tanpa menyimpan')
    mode.add_argument('--run',action='store_true',help='Audit, hitung OOF, simpan dan kunci hasil')
    mode.add_argument('--verify',action='store_true',help='Periksa hash hasil terkunci dan audit ulang')
    args=ap.parse_args()
    if args.verify:
        verify()
        return
    folds,cmp_df,preds,report=compute_all()
    show(cmp_df)
    if args.check:
        print('\nPASS_E7_OOF_PREFLIGHT_20_RUNS | no outputs written')
        return
    write_all(folds,cmp_df,preds,report)

if __name__=='__main__':
    main()
