from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
FOLD_ROOT = ROOT / 'data/manifests/experiment4_fixedtest257_dev1411_5fold_seed42_v1'
TRAIN_ROOT = ROOT / 'output/training'
OUT = ROOT / 'output/cross_validation/04_oof_model_selection_v1'
CONFIG = SRC / '02_training_config_e4v2_LOCKED.json'
AUDIT = ROOT / 'audits/02_dataloader_validation/02_e4v2_dataloader_validation_report.json'
LOCK = FOLD_ROOT / '01_e4_cv85_fold_lock_report.json'
DEV = FOLD_ROOT / 'development_1411_5fold_LOCKED.csv'
CONFIG_SHA = 'a9b9d2eda5e260e20802144363768f3ba18b467b017ff0e5222e08b6504c0b63'
MODEL_COMMIT = '110cbc18d5133582e320b43d53bf5c44e410c936'
KEY = ['source_archive','archive_member_path']
IDENTITY = KEY + ['patient_id','y_true','cv_fold']
MODELS = [('frozen','raw'), ('frozen','clahe'), ('lora','raw'), ('lora','clahe')]
STATUS = 'PASS_E4V2_OOF_MODEL_SELECTION_LOCKED'
SELECTION_FILE = '04_e4v2_oof_model_selection_LOCKED.json'
OUTPUT_FILES = ['04_e4v2_fold_metrics_20runs.csv','04_e4v2_oof_comparison_4models.csv'] + [
    f'04_e4v2_oof_predictions_{method}_{variant}.csv' for method,variant in MODELS
]


def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def jread(p):
    need(Path(p).is_file(),f'Missing JSON: {p}')
    return json.loads(Path(p).read_text(encoding='utf-8'))


def cread(p):
    need(Path(p).is_file(),f'Missing CSV: {p}')
    return pd.read_csv(p,dtype=str,keep_default_na=False,low_memory=False)


def eq_num(a,b,where,eps=1e-8):
    need(math.isfinite(float(a)) and math.isfinite(float(b)) and
         math.isclose(float(a),float(b),abs_tol=eps,rel_tol=eps),
         f'Numerical discrepancy ({where}): {a} vs {b}')


def ids(df):
    return set(map(tuple,df[KEY].to_numpy()))


def metrics(y,probs):
    y=np.asarray(y,dtype=int);p=np.asarray(probs,dtype=float)
    need(len(y)>0 and len(y)==len(p) and np.isfinite(p).all() and
         ((p>=0)&(p<=1)).all() and set(y)=={0,1},'Invalid probabilities/labels')
    prediction=(p>=.5).astype(int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,prediction,labels=[0,1]).ravel())
    return {
        'auroc':float(roc_auc_score(y,p)),
        'accuracy':float(np.mean(y==prediction)),
        'macro_f1':float(f1_score(y,prediction,average='macro',zero_division=0)),
        'balanced_accuracy':float(balanced_accuracy_score(y,prediction)),
        'sensitivity_dr':float(tp/(tp+fn)),
        'specificity_ds':float(tn/(tn+fp)),
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'threshold':.5,
    }


def sources():
    need(CONFIG.is_file() and digest(CONFIG)==CONFIG_SHA,'E4v2 locked training config SHA256 mismatch')
    cfg=jread(CONFIG)
    need(cfg.get('status')=='LOCKED_PROTOCOL_E4V2_ALIGNED_WITH_E6_2026-10-23' and
         cfg.get('experiment')=='experiment4v2' and cfg.get('folds')==5 and
         cfg.get('primary_model_selection_after_cv')=='pooled_OOF_AUROC' and
         cfg.get('final_inference')=='five_checkpoint_probability_ensemble_on_original_E4_test',
         'E4v2 model selection protocol changed')
    lock=jread(LOCK);audit=jread(AUDIT)
    need(lock.get('status')=='PASS_E4_CV85_FOLDS_LOCKED' and
         lock.get('no_patient_leakage') is True and
         lock.get('same_test_patients_as_original_e4') is True and
         lock.get('each_development_image_oof_once') is True and
         lock.get('each_development_patient_oof_once') is True and
         lock.get('test_pixel_accessed') is False,'E4v2 fold lock failed')
    need(digest(LOCK)==cfg['locked_e4v2_fold_report_sha256'] and
         digest(DEV)==cfg['locked_e4v2_development_manifest_sha256'],
         'E4v2 fold report/development manifest changed')
    need(audit.get('status')=='PASS_E4V2_DATALOADER_VALIDATION' and
         audit.get('checked_combinations')==20 and
         audit.get('source_fold_report_sha256')==digest(LOCK) and
         audit.get('test_pixel_accessed') is False and
         audit.get('model_training_performed') is False,
         'E4v2 DataLoader audit is not verified')
    need(lock['development']['images']==1411 and lock['development']['patients']==1091 and
         lock['test']['images']==257 and lock['test']['patients']==193 and
         lock['development']['ds_images']==478 and lock['development']['dr_images']==933,
         'Locked E4v2 cohort counts changed')
    for rel, hash_expected in lock['output_sha256'].items():
        relative=Path(rel)
        need(not relative.is_absolute() and '..' not in relative.parts,'Unsafe locked relative path')
        location=FOLD_ROOT/relative
        need(location.is_file() and digest(location)==hash_expected,
             f'Locked fold artifact mismatch: {rel}')
    dev=cread(DEV)
    need(set(IDENTITY+['e4_original_split']).issubset(dev.columns),
         'E4v2 development manifest columns missing')
    need(len(dev)==1411 and dev.patient_id.nunique()==1091 and
         dev.y_true.value_counts().to_dict()=={'0':478,'1':933} and
         dev.e4_original_split.isin(['train','validation']).all() and
         dev.cv_fold.isin([str(i) for i in range(1,6)]).all() and
         not dev[KEY].duplicated().any() and
         dev.groupby('patient_id').cv_fold.nunique().eq(1).all() and
         dev.groupby('patient_id').y_true.nunique().eq(1).all(),
         'E4v2 development data invalid or changed')
    for fold in range(1,6):
        val=cread(FOLD_ROOT/f'fold_{fold}/validation.csv')
        expected=dev.loc[dev.cv_fold==str(fold),IDENTITY]
        need(set(IDENTITY+['split']).issubset(val.columns) and
             val.split.eq('validation').all() and len(expected)==len(val) and
             not val[KEY].duplicated().any() and ids(val)==ids(expected),
             f'Fold {fold} validation images invalid')
        joined=expected.merge(val[IDENTITY],on=KEY,suffixes=('_locked','_fold'),validate='one_to_one')
        for attr in ('patient_id','y_true','cv_fold'):
            need(joined[f'{attr}_locked'].eq(joined[f'{attr}_fold']).all(),
                 f'Fold {fold} {attr} mismatch')
    hashes={
        'training_config':digest(CONFIG),
        'fold_lock_report':digest(LOCK),
        'development_manifest':digest(DEV),
        'dataloader_report':digest(AUDIT),
        'e4_original_fixed_test_reference_only_sha256':lock['source_sha256']['split'],
    }
    return cfg,lock,audit,dev,hashes


def inspect(method,variant,fold,cfg,lock,dev,hashes):
    name=f'{method}_{variant}'
    folder=TRAIN_ROOT/f'{name}_fold_{fold}'
    filenames=['run_summary.json','run_config.json','training_history.csv',
               'best_validation_metrics.json','best_validation_predictions.csv','best_classifier.pt']
    if method=='lora':
        filenames+=['best_adapter/adapter_model.safetensors','best_adapter/adapter_config.json']
    for filename in filenames:
        p=folder/filename
        need(p.is_file() and p.stat().st_size>0,
             f'Missing/incomplete artifact: {name} fold {fold}: {filename}')
    summary=jread(folder/'run_summary.json')
    run=jread(folder/'run_config.json')
    reported=jread(folder/'best_validation_metrics.json')
    hist=cread(folder/'training_history.csv')
    pred=cread(folder/'best_validation_predictions.csv')
    need(summary.get('status')=='PASS_E4V2_FOLD_TRAINING_COMPLETE' and
         summary.get('experiment')=='experiment4v2' and
         summary.get('method')==method and summary.get('variant')==variant and
         summary.get('fold')==fold and summary.get('five_fold_cv') is True and
         summary.get('test_pixel_read') is False and summary.get('test_inference') is False,
         f'Run summary mismatched: {name} F{fold}')
    need(run.get('experiment')=='experiment4v2' and run.get('method')==method and
         run.get('variant')==variant and run.get('fold')==fold and
         run.get('test_data_used') is False and run.get('seed')==42 and
         run.get('max_epochs')==30 and run.get('patience')==5,
         f'Run config mismatch: {name} F{fold}')
    for field,val in [('min_delta',.001),('lr_head',.001),('weight_decay',.0001)]:
        eq_num(run.get(field),val,f'{name} F{fold} {field}')
    if method=='frozen':
        need(run.get('head_batch_size')==64,f'Frozen head batch mismatch: {name} F{fold}')
    else:
        need(run.get('physical_batch')==1 and run.get('gradient_accumulation')==16 and
             run.get('effective_batch')==16 and
             run.get('lora_layers_0_based')==[8,9,10,11] and
             run.get('lora_projections')==['query','value'] and
             run.get('lora_r')==8 and run.get('lora_alpha')==16,
             f'LoRA architecture changed: {name} F{fold}')
        for field,val in [('lr_lora',.0001),('lora_dropout',.05)]:
            eq_num(run.get(field),val,f'{name} F{fold} {field}')
    fp=run.get('training_fingerprints',{})
    need(summary.get('fingerprints')==fp and
         fp.get('training_config')==hashes['training_config'] and
         fp.get('stage1_e4v2_fold_report')==hashes['fold_lock_report'] and
         fp.get('e4v2_dev_locked')==hashes['development_manifest'] and
         fp.get('stage2_dataloader_sha256')==hashes['dataloader_report'] and
         fp.get('fold')==fold and
         fp.get('fold_validation_manifest')==lock['output_sha256'][f'fold_{fold}/validation.csv'] and
         fp.get('fold_train_manifest')==lock['output_sha256'][f'fold_{fold}/train.csv'],
         f'Run provenance mismatch: {name} F{fold}')
    expected=dev.loc[dev.cv_fold==str(fold)]
    need(summary.get('validation_images')==len(expected) and
         summary.get('train_images')==1411-len(expected) and
         fp.get('validation_images')==len(expected) and
         fp.get('train_images')==1411-len(expected),
         f'Fold image counts changed: {name} F{fold}')
    need(summary.get('model_id')=='microsoft/rad-dino' and
         summary.get('model_commit')==MODEL_COMMIT and
         run.get('model_commit')==MODEL_COMMIT,
         f'RAD-DINO commit mismatch: {name} F{fold}')
    need(set(IDENTITY+['p_dr','pred_0_5','split']).issubset(pred.columns) and
         len(pred)==len(expected) and not pred[KEY].duplicated().any() and
         pred.split.eq('validation').all() and pred.cv_fold.eq(str(fold)).all() and
         ids(pred)==ids(expected), f'Validation predictions do not match {name} F{fold}')
    merged=expected[IDENTITY].merge(pred[IDENTITY],on=KEY,
                                    suffixes=('_locked','_pred'),validate='one_to_one')
    for field in ('patient_id','y_true','cv_fold'):
        need(merged[f'{field}_locked'].eq(merged[f'{field}_pred']).all(),
             f'Prediction identity {field} changed: {name} F{fold}')
    need(pred.y_true.isin(['0','1']).all() and pred.pred_0_5.isin(['0','1']).all(),
         f'Prediction labels invalid: {name} F{fold}')
    p=pd.to_numeric(pred.p_dr,errors='coerce').to_numpy(dtype=float)
    need(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and
         np.array_equal((p>=.5).astype(int),pred.pred_0_5.astype(int).to_numpy()),
         f'Prediction probabilities invalid: {name} F{fold}')
    met=metrics(pred.y_true.astype(int).to_numpy(),p)
    best_epoch=int(summary['best_epoch'])
    need(best_epoch==int(reported['best_epoch']) and
         1<=best_epoch<=int(summary['last_epoch'])<=30 and
         int(summary['epochs_recorded'])==int(summary['last_epoch'])==len(hist) and
         hist.epoch.tolist()==[str(i) for i in range(1,len(hist)+1)],
         f'Run epoch history invalid: {name} F{fold}')
    eq_num(hist.iloc[best_epoch-1]['val_auroc'],met['auroc'],f'History AUROC {name} F{fold}')
    eq_num(summary['best_validation_auroc'],met['auroc'],f'Run AUROC {name} F{fold}')
    for key,value in met.items():
        need(key in reported, f'Best-validation metric missing: {key} ({name} F{fold})')
        if key in ('tn','fp','fn','tp'):
            need(int(reported[key])==value,f'Confusion matrix mismatch: {name} F{fold} {key}')
        else:
            eq_num(reported[key],value,f'Metric {key} {name} F{fold}')
    row={
        'model':name,'method':method,'variant':variant,'fold':fold,
        'n_validation_images':len(pred),'n_validation_patients':pred.patient_id.nunique(),
        'n_ds':int(pred.y_true.eq('0').sum()),'n_dr':int(pred.y_true.eq('1').sum()),
        'best_epoch':best_epoch,'last_epoch':int(summary['last_epoch']),**met,
    }
    out_pred=pred[IDENTITY+['p_dr','pred_0_5']].copy()
    out_pred['p_dr']=p
    out_pred['model']=name
    return row,out_pred,{f:digest(folder/f) for f in filenames}


def compute_all():
    cfg,lock,audit,dev,hashes=sources()
    fold_rows=[];model_rows=[];predictions={};provenance={}
    for method,variant in MODELS:
        name=f'{method}_{variant}';all_preds=[];fold_metrics=[]
        for fold in range(1,6):
            row,pred,artifact_hash=inspect(method,variant,fold,cfg,lock,dev,hashes)
            fold_rows.append(row);fold_metrics.append(row);all_preds.append(pred)
            provenance[f'{name}_fold_{fold}']=artifact_hash
        pooled=pd.concat(all_preds,ignore_index=True)
        need(len(pooled)==1411 and pooled.patient_id.nunique()==1091 and
             ids(pooled)==ids(dev) and not pooled[KEY].duplicated().any(),
             f'OOF coverage not complete: {name}')
        joint=dev[IDENTITY].merge(pooled[IDENTITY],on=KEY,
                                  suffixes=('_locked','_oof'),validate='one_to_one')
        for col in ('patient_id','y_true','cv_fold'):
            need(joint[f'{col}_locked'].eq(joint[f'{col}_oof']).all(),
                 f'OOF identity mismatch: {name} {col}')
        pooled=pooled.sort_values(KEY,kind='stable').reset_index(drop=True)
        overall=metrics(pooled.y_true.astype(int).to_numpy(),pooled.p_dr.to_numpy(dtype=float))
        aurocs=np.array([x['auroc'] for x in fold_metrics],dtype=float)
        model_rows.append({
            'model':name,'method':method,'variant':variant,
            'oof_images':1411,'oof_patients':1091,'n_folds':5,
            'fold_auroc_mean':float(np.mean(aurocs)),
            'fold_auroc_sd_sample':float(np.std(aurocs,ddof=1)),
            'fold_auroc_min':float(np.min(aurocs)),
            'fold_auroc_max':float(np.max(aurocs)),
            'oof_auroc':overall['auroc'],
            'oof_macro_f1':overall['macro_f1'],
            'oof_balanced_accuracy':overall['balanced_accuracy'],
            'oof_sensitivity_dr':overall['sensitivity_dr'],
            'oof_specificity_ds':overall['specificity_ds'],
            'oof_accuracy':overall['accuracy'],
            'tn':overall['tn'],'fp':overall['fp'],'fn':overall['fn'],'tp':overall['tp'],
            'threshold':.5,
        })
        predictions[name]=pooled
    folds=pd.DataFrame(fold_rows)
    summary=pd.DataFrame(model_rows).sort_values(
        ['oof_auroc','fold_auroc_mean','model'],
        ascending=[False,False,True],kind='stable').reset_index(drop=True)
    winner=str(summary.iloc[0]['model'])
    report={
        'status':STATUS,'experiment':'experiment4v2',
        'protocol':'Original E4 fixed test 257 images (15.41%), 5-fold CV on 1411 development images (84.59%)',
        'method':'Reaggregate existing best-fold validation probabilities; NO new training/inference',
        'selection_criterion':'maximum pooled image-level OOF AUROC',
        'tie_breakers':['higher mean fold AUROC','alphabetic model name'],
        'primary_model':winner,'primary_model_oof_auroc':float(summary.iloc[0]['oof_auroc']),
        'primary_selected_before_e4v2_test':True,
        'n_development_images':1411,'n_development_patients':1091,
        'n_test_images_REFERENCE_ONLY':257,'n_models':4,'n_training_runs':20,
        'source_sha256':hashes,'run_artifact_sha256':provenance,
        'models':summary.to_dict('records'),
        'test_pixel_read':False,'test_predictions_read':False,
        'test_inference_performed':False,'new_training_performed':False,
        'notes':[
            'OOF outcomes are IMAGE level; patients are grouped across folds but no patient-level metric calculated.',
            'The same fold validation data selected checkpoint and supplied OOF metric; may be optimistic; not nested CV.',
            'Fold AUROC SD is descriptive, using sample SD ddof=1.',
            'Original E4 testing was previously exposed. E4v2 test reuse is exploratory.',
            'Original E4 and E6 testing involve different patients and imbalanced versus balanced cohorts.',
            'Final 5-checkpoint ensemble evaluation on original E4 holdout is a separate next stage.',
        ],
    }
    return folds,summary,predictions,report


def write_all(folds,summary,predictions,report):
    need(not OUT.exists(),f'Refusing overwrite of OOF output folder: {OUT}; use --verify')
    OUT.mkdir(parents=True,exist_ok=False)
    folds.to_csv(OUT/OUTPUT_FILES[0],index=False)
    summary.to_csv(OUT/OUTPUT_FILES[1],index=False)
    for method,variant in MODELS:
        name=f'{method}_{variant}'
        predictions[name].to_csv(OUT/f'04_e4v2_oof_predictions_{name}.csv',index=False)
    report['output_sha256']={filename:digest(OUT/filename) for filename in OUTPUT_FILES}
    (OUT/SELECTION_FILE).write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/(SELECTION_FILE+'.sha256')).write_text(
        digest(OUT/SELECTION_FILE)+'  '+SELECTION_FILE+'\n',encoding='utf-8')


def verify_all():
    report=jread(OUT/SELECTION_FILE)
    need(report.get('status')==STATUS,'OOF selection status not locked')
    seal=(OUT/(SELECTION_FILE+'.sha256')).read_text(encoding='utf-8').strip()
    need(seal==digest(OUT/SELECTION_FILE)+'  '+SELECTION_FILE,
         'OOF lock checksum mismatch')
    for file in OUTPUT_FILES:
        need(file in report['output_sha256'] and
             digest(OUT/file)==report['output_sha256'][file],
             f'OOF output tampered: {file}')
    _,cur_summary,_,actual=compute_all()
    need(actual['source_sha256']==report['source_sha256'] and
         actual['run_artifact_sha256']==report['run_artifact_sha256'] and
         actual['models']==report['models'] and
         actual['primary_model']==report['primary_model'],
         'OOF sources/selection changed after locking')
    need(str(cur_summary.iloc[0]['model'])==report['primary_model'],'OOF winning model mismatch')
    print(f'PASS_E4V2_OOF_MODEL_SELECTION_VERIFIED | primary={report["primary_model"]} | test not accessed')


def show(summary):
    columns=['model','fold_auroc_mean','fold_auroc_sd_sample','oof_auroc',
             'oof_macro_f1','oof_balanced_accuracy','oof_sensitivity_dr','oof_specificity_ds']
    print('\nE4V2 5-FOLD OOF EVALUATION | 1411 DEVELOPMENT IMAGES | TEST NOT READ')
    print(summary[columns].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('PRIMARY (pooled OOF AUROC):',summary.iloc[0]['model'])
    print('NOTE: checkpoint selection used the same fold validation data; OOF can be optimistic.')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    g=a.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true')
    g.add_argument('--run',action='store_true')
    g.add_argument('--verify',action='store_true')
    args=a.parse_args()
    if args.verify:
        verify_all();return
    folds,summary,preds,report=compute_all()
    show(summary)
    if args.check:
        print('\nPASS_E4V2_OOF_PREFLIGHT_20_RUNS | no output written')
        return
    write_all(folds,summary,preds,report)
    print('\n'+STATUS+' | output:',OUT)
    verify_all()


if __name__=='__main__':
    try:main()
    except Exception as err:
        print(f'FAIL_E4V2_OOF_STAGE4: {type(err).__name__}: {err}',file=sys.stderr,flush=True)
        raise SystemExit(1)
