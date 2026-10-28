from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import e6_training_common as C

ROOT=C.ROOT
OOF=ROOT/'output/cross_validation/05_oof_model_selection_v1'
OOF_LOCK=OOF/'05_e6_oof_model_selection_LOCKED.json'
OOF_LOCK_SHA='5a96a427b25156a9252d8af3d13e7a05dbe71fff2a3f733afaa1f2cb8f32041b'
OUTPUT=ROOT/'output/testing/06_e6_five_fold_ensemble_holdout_v1'
STAGING=OUTPUT.with_name(OUTPUT.name+'_staging')
TRAINING=ROOT/'output/training'
KEY=['source_archive','archive_member_path']
ID=KEY+['patient_id','y_true']
MODELS=(('M1','frozen','raw'),('M2','frozen','clahe'),
        ('M3','lora','raw'),('M4','lora','clahe'))
OUT_STATUS='PASS_E6_FIXED_HOLDOUT_5FOLD_ENSEMBLE'


def require(ok,message): C.need(ok,message)


def jread(p): return C.read_json(p)


def expected_key(df):
    return (df.source_archive+'::'+df.archive_member_path).tolist()


def model_name(method,variant): return f'{method}_{variant}'


def fold_filename(method,variant,fold):
    return f'06_test_predictions_{method}_{variant}_fold_{fold}.csv'


def ensemble_filename(method,variant):
    return f'06_test_ensemble_predictions_{method}_{variant}.csv'


def test_metadata(config):
    """Test manifest identities, labels and PNG paths; no Image.open here."""
    require(C.sha(C.TEST_SOURCE)==config['e5_test_manifest_sha256'],
            'E5 test manifest changed')
    require(C.sha(C.STAGE5/'processed_manifest.csv')==
            config['expected_hashes']['e5_processed_manifest_sha256'],
            'E5 processed manifest changed')
    test=C.read_csv(C.TEST_SOURCE)
    manifest=C.read_csv(C.STAGE5/'processed_manifest.csv')
    require('split' in manifest and set(ID+['split','raw_path','clahe_path']).issubset(manifest.columns),
            'E5 processed manifest columns incompatible')
    rows=manifest.loc[manifest.split.eq('test')].copy()
    require(test.split.eq('test').all() and len(test)==333 and len(rows)==333,
            'Unexpected fixed E5 holdout size/split')
    require(not test[KEY].duplicated().any() and not rows[KEY].duplicated().any(),
            'Duplicate E5 holdout images')
    test=test[ID].sort_values(KEY).reset_index(drop=True)
    rows=rows.sort_values(KEY).reset_index(drop=True)
    require(rows[ID].equals(test[ID]),'Processed test identities/labels disagree with E5 test.csv')
    require(rows.patient_id.nunique()==259 and rows.y_true.value_counts().to_dict()=={'0':166,'1':167},
            'Unexpected E5 test patients or class counts')
    development=C.read_csv(C.FOLDS/'e6_development_1887_5fold_LOCKED.csv')
    require(len(development)==1887 and development.patient_id.nunique()==1471,
            'Development reference unexpectedly changed')
    require(not set(rows.patient_id).intersection(development.patient_id),
            'E6 development patient overlaps E5 test patient')
    require(not set(map(tuple,rows[KEY].to_numpy())).intersection(
            set(map(tuple,development[KEY].to_numpy()))),
            'E6 development image overlaps E5 test image')
    # Metadata-only path existence; no pixel reads.
    for variant in ('raw','clahe'):
        require(rows[variant+'_path'].nunique()==333,
                f'Duplicate {variant} PNG path')
        for path in rows[variant+'_path']:
            require(C.safe_png_path(variant,path).is_file(),f'Missing test PNG: {path}')
    return rows


def preflight():
    require(OOF_LOCK.is_file(),'Stage 5 OOF model selection lock not found')
    require(C.sha(OOF_LOCK)==OOF_LOCK_SHA,
            'E6 Stage 5 selection LOCK has changed since independently audited upload')
    seal=(OOF/(OOF_LOCK.name+'.sha256')).read_text(encoding='utf-8').strip()
    require(seal==OOF_LOCK_SHA+'  '+OOF_LOCK.name,
            'Stage 5 model selection sidecar SHA256 mismatch')
    lock=jread(OOF_LOCK)
    require(lock.get('status')=='PASS_E6_OOF_MODEL_SELECTION_LOCKED' and
            lock.get('primary_model')=='lora_clahe' and
            lock.get('primary_selected_before_e6_testing') is True and
            lock.get('test_pixel_read') is False and
            lock.get('test_inference_performed') is False and
            lock.get('n_training_runs')==20 and lock.get('n_models')==4,
            'Stage 5 did not lock M4 from the 20 internal fold runs')
    require(abs(float(lock['primary_model_auroc'])-0.7178951282419973)<1e-12,
            'Stage 5 primary AUROC changed')
    cfg=C.load_config()
    require(cfg['primary_model_selection_after_cv']=='out_of_fold_AUROC' and
            cfg['final_inference']=='five_checkpoint_probability_ensemble',
            'E6 inference protocol changed')
    require(lock['source_sha256']['training_config']==C.sha(C.CONFIG_PATH) and
            lock['source_sha256']['fold_lock_report']==C.sha(C.FOLDS/'02_e6_5fold_lock_report.json') and
            lock['source_sha256']['dataloader_audit']==C.sha(C.AUDIT) and
            lock['source_sha256']['fold_development_manifest']==C.sha(
                C.FOLDS/'e6_development_1887_5fold_LOCKED.csv') and
            lock['source_sha256']['fixed_test_manifest_reference_SHA256_NO_TEST_READ']==C.sha(C.TEST_SOURCE),
            'Stage 5 input provenance changed')
    for fname,digest in lock['output_sha256'].items():
        require(C.sha(OOF/fname)==digest,f'Stage 5 OOF output modified: {fname}')
    require([x['model'] for x in lock['models']]==
            ['lora_clahe','frozen_clahe','lora_raw','frozen_raw'],
            'Stage 5 rankings changed')
    run_info={}
    for code,method,variant in MODELS:
        for fold in range(1,6):
            run_name=f'{method}_{variant}_fold_{fold}'
            folder=TRAINING/run_name
            provenance=lock['run_artifact_sha256'].get(run_name)
            require(provenance is not None,f'No Stage 5 run provenance: {run_name}')
            for relative,hash_expected in provenance.items():
                require(C.sha(folder/relative)==hash_expected,
                        f'Best checkpoint/training artifact modified: {run_name}/{relative}')
            summary=jread(folder/'run_summary.json')
            run=jread(folder/'run_config.json')
            require(summary.get('status')=='PASS_E6_FOLD_TRAINING_COMPLETE' and
                    summary.get('fold')==fold and summary.get('method')==method and
                    summary.get('variant')==variant and
                    run.get('fold')==fold and run.get('test_data_used') is False and
                    summary.get('fingerprints')==run.get('training_fingerprints') and
                    summary.get('model_commit')==C.MODEL_COMMIT_EXPECTED,
                    f'Run summary/config inconsistency: {run_name}')
            require(1<=summary['best_epoch']<=summary['last_epoch']<=30,
                    f'Checkpoint epoch invalid: {run_name}')
            run_info[run_name]=(folder,summary,run)
    rows=test_metadata(cfg)
    print('PASS_E6_STAGE6_PREFLIGHT | 20 BEST checkpoints hashes verified',flush=True)
    print('E5 fixed test 333 images / 259 patients / DS=166, DR=167; no test pixels opened',flush=True)
    print('Primary model LOCKED BEFORE TEST: M4 = LoRA CLAHE (OOF AUROC 0.717895)',flush=True)
    return lock,cfg,rows,run_info


class TestPNG(Dataset):
    def __init__(self, rows, variant, processor):
        require(len(rows)==333 and rows.split.eq('test').all() and variant in ('raw','clahe'),
                'Explicit fixed-test loader only')
        self.rows=rows.reset_index(drop=True);self.variant=variant;self.processor=processor
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]
        path=C.safe_png_path(self.variant,row[self.variant+'_path'])
        with Image.open(path) as im:
            require(im.mode=='L' and im.size==(518,518),f'Invalid fixed-test PNG: {path}')
            rgb=im.convert('RGB')
            tensor=self.processor(images=rgb,do_resize=False,do_center_crop=False,
                                  return_tensors='pt')['pixel_values'].squeeze(0)
        require(tuple(tensor.shape)==(3,518,518) and torch.isfinite(tensor).all().item(),
                'Test pixel tensor invalid')
        return tensor,torch.tensor(int(row.y_true),dtype=torch.long),str(row.patient_id),(
            row.source_archive+'::'+row.archive_member_path)


def head_from_best(folder,summary,run,device):
    state=C.torch_load_trusted(folder/'best_classifier.pt')
    fp=run['training_fingerprints']
    require(state.get('fingerprints')==fp and
            state.get('best_epoch')==summary['best_epoch'] and
            abs(float(state.get('best_val_auroc',-1))-
                float(summary['best_validation_auroc']))<=1e-9 and
            state.get('model_commit')==summary['model_commit'],
            'Classifier checkpoint does not match locked best validation epoch')
    head=nn.Linear(768,2)
    head.load_state_dict(state['classifier_state_dict'],strict=True)
    return head.to(device).eval()


def inference_fold(rows,method,variant,folder,summary,run,processor,device,
                   offline,bs,progress):
    backbone=C.backbone_new(offline)
    require(getattr(backbone.config,'_commit_hash',None)==summary['model_commit'],
            'Backbone revision mismatch during final test')
    if method=='lora':
        from peft import PeftModel
        backbone=PeftModel.from_pretrained(backbone,str(folder/'best_adapter'),is_trainable=False)
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone=backbone.to(device).eval()
    head=head_from_best(folder,summary,run,device)
    loader=DataLoader(TestPNG(rows,variant,processor),batch_size=bs if method=='frozen' else 1,
                      shuffle=False,drop_last=False,num_workers=0,pin_memory=True)
    y_vals=[];keys=[];probs=[];seen=0
    try:
        with torch.inference_mode():
            for x,y,patients,identities in loader:
                with torch.cuda.amp.autocast(enabled=(method=='lora')):
                    z=backbone(pixel_values=x.to(device,non_blocking=True)).last_hidden_state[:,0,:]
                    logits=head(z).float()
                require(tuple(logits.shape)==(len(y),2) and
                        torch.isfinite(logits).all().item(),
                        'Invalid logits in final holdout inference')
                p=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
                require(np.isfinite(p).all() and np.logical_and(p>=0,p<=1).all(),
                        'Invalid test probabilities')
                probs.extend(p.tolist());keys.extend(list(identities));y_vals.extend(y.tolist())
                seen+=len(y)
                if seen%progress==0 or seen==len(rows):
                    print(f'  {method}/{variant}: {seen}/333 test images',flush=True)
    finally:
        del backbone,head,loader
        gc.collect()
        torch.cuda.empty_cache()
    require(seen==333 and keys==expected_key(rows) and
            y_vals==rows.y_true.astype(int).tolist(),
            'Test inference image identity/order/label mismatch')
    return np.asarray(probs,dtype=float)


def fold_table(rows,prob,fold):
    result=rows[ID].copy()
    result['fold']=fold;result['p_dr']=prob
    return result


def load_fold_prediction(path,rows,fold):
    df=C.read_csv(path)
    require(list(df.columns)==ID+['fold','p_dr'] and len(df)==333 and
            df.fold.eq(str(fold)).all() and df[ID].equals(rows[ID]),
            f'Invalid or incompatible saved fold prediction: {path}')
    p=pd.to_numeric(df.p_dr,errors='coerce').to_numpy(dtype=float)
    require(np.isfinite(p).all() and np.logical_and(p>=0,p<=1).all(),
            f'Non-finite or outside [0,1] fold probabilities: {path}')
    return p


def ensemble_table(rows,probs):
    p=np.mean(np.stack(probs,axis=0),axis=0)
    require(tuple(p.shape)==(333,) and np.isfinite(p).all() and
            np.logical_and(p>=0,p<=1).all(),'Ensemble probabilities invalid')
    df=rows[ID].copy()
    df['p_dr_mean_5fold']=p
    df['pred_0_5']=(p>=.5).astype(int)
    df['correct']=(df.pred_0_5.to_numpy()==rows.y_true.astype(int).to_numpy()).astype(int)
    return df


def read_ensemble(path,rows,ps):
    df=C.read_csv(path)
    require(list(df.columns)==ID+['p_dr_mean_5fold','pred_0_5','correct'] and
            len(df)==333 and df[ID].equals(rows[ID]),
            f'Ensemble predictions inconsistent: {path}')
    expected=ensemble_table(rows,ps)
    p=pd.to_numeric(df.p_dr_mean_5fold,errors='coerce').to_numpy(dtype=float)
    require(np.allclose(p,expected.p_dr_mean_5fold.to_numpy(),rtol=0,atol=1e-14) and
            df.pred_0_5.astype(int).tolist()==expected.pred_0_5.tolist() and
            df.correct.astype(int).tolist()==expected.correct.tolist(),
            f'Ensemble is not exact average of five fold probabilities: {path}')
    return p


def compute_saved_results(folder,rows):
    results=[];comparison=[];pred_sha={}
    for code,method,variant in MODELS:
        probs=[]
        for fold in range(1,6):
            pth=folder/fold_filename(method,variant,fold)
            require(pth.is_file(),f'Missing fold test prediction {pth}')
            probs.append(load_fold_prediction(pth,rows,fold))
            pred_sha[pth.name]=C.sha(pth)
        en=folder/ensemble_filename(method,variant)
        require(en.is_file(),f'Missing ensemble predictions {en}')
        p=read_ensemble(en,rows,probs)
        pred_sha[en.name]=C.sha(en)
        metrics=C.classification_metrics(rows.y_true.astype(int).to_numpy(),p)
        require(metrics['tn']+metrics['fp']==166 and
                metrics['fn']+metrics['tp']==167,'Holdout confusion matrix counts invalid')
        results.append({'model_code':code,'model':model_name(method,variant),
                        'method':method,'variant':variant,
                        'primary_preselected':code=='M4','n_folds':5,
                        'n_test_images':333,'n_test_patients':259,**metrics})
        print(f'  {code} ensemble: AUROC={metrics["auroc"]:.6f} '
              f'MacroF1={metrics["macro_f1"]:.6f} '
              f'BalAcc={metrics["balanced_accuracy"]:.6f} '
              f'SensDR={metrics["sensitivity_dr"]:.6f} '
              f'SpecDS={metrics["specificity_ds"]:.6f}',flush=True)
    return pd.DataFrame(results),pred_sha


def save_comparison_and_report(folder,rows,lock):
    cmp, hashes=compute_saved_results(folder,rows)
    comparison_path=folder/'06_e6_ensemble_test_comparison.csv'
    C.atomic_csv(comparison_path,cmp)
    hashes[comparison_path.name]=C.sha(comparison_path)
    primary=cmp.loc[cmp.model_code=='M4'].iloc[0].to_dict()
    report={'status':OUT_STATUS,'experiment':'experiment6',
            'evaluation_protocol':'fixed E5 holdout 15%, five best-fold checkpoint probability ensembles',
            'primary_model':'lora_clahe','primary_model_code':'M4',
            'primary_selection_criterion':'pooled OOF image-level AUROC',
            'primary_selection_locked_before_testing':True,
            'primary_model_oof_auroc':float(lock['primary_model_auroc']),
            'primary_model_test_metrics':primary,
            'test_set_is_reused_e5_holdout_exploratory':True,
            'test_based_model_selection':False,
            'threshold':.5,
            'test_cohort':{'images':333,'patients':259,'DS-TB':166,'DR-TB':167},
            'model_comparison':cmp.to_dict('records'),
            'test_manifest_sha256':C.sha(C.TEST_SOURCE),
            'stage5_model_selection_lock_sha256':C.sha(OOF_LOCK),
            'training_during_test':False,
            'inference':{'aggregation':'unweighted arithmetic mean of five best fold probabilities',
                         'checkpoints_per_configuration':5,
                         'prediction_unit':'image',
                         'fold_checkpoint_selection':'fold_validation_AUROC',
                         'decision_threshold':.5},
            'output_sha256':hashes}
    C.atomic_json(folder/'06_e6_ensemble_holdout_test_LOCKED.json',report)
    seal=folder/'06_e6_ensemble_holdout_test_LOCKED.json.sha256'
    seal.write_text(C.sha(folder/'06_e6_ensemble_holdout_test_LOCKED.json')+'  '+
                    '06_e6_ensemble_holdout_test_LOCKED.json\n',encoding='utf-8')
    return report


def verify(folder,rows,lock):
    reportfile=folder/'06_e6_ensemble_holdout_test_LOCKED.json'
    report=jread(reportfile)
    require(report.get('status')==OUT_STATUS and
            report.get('primary_model')==lock['primary_model'] and
            report.get('stage5_model_selection_lock_sha256')==C.sha(OOF_LOCK) and
            report.get('test_manifest_sha256')==C.sha(C.TEST_SOURCE),
            'Final test report provenance/status incorrect')
    require((folder/(reportfile.name+'.sha256')).read_text(encoding='utf-8').strip()==
            C.sha(reportfile)+'  '+reportfile.name,'Final report seal SHA256 mismatch')
    cmp,hashes=compute_saved_results(folder,rows)
    for fname,digest in report['output_sha256'].items():
        require(C.sha(folder/fname)==digest,f'Test artifact modified: {fname}')
    require(set(hashes).issubset(report['output_sha256']) and
            cmp.to_dict('records')==report['model_comparison'],
            'Final test metrics differ from locked predictions')
    cpath=folder/'06_e6_ensemble_test_comparison.csv'
    cmpfile=C.read_csv(cpath)
    require(len(cmpfile)==4 and cmpfile.model.tolist()==[f'{m}_{v}' for _,m,v in MODELS],
            'Comparison row order or labels invalid')
    for _,r in cmp.iterrows():
        stored=cmpfile.loc[cmpfile.model==r['model']].iloc[0]
        for k in ('auroc','macro_f1','balanced_accuracy','sensitivity_dr','specificity_ds',
                  'accuracy','threshold'):
            require(abs(float(stored[k])-float(r[k]))<1e-12,
                    f'CSV metric changed: {r["model"]}/{k}')
    print(OUT_STATUS+'_VERIFIED | M4 remains primary | no new inference',flush=True)


def run_test(a,lock,rows,infos):
    require(torch.cuda.is_available(),'GPU CUDA is required for E6 final inference')
    if a.run_test:
        require(not OUTPUT.exists() and not STAGING.exists(),
                f'Previous test output/staging exists; refuse overwrite: {OUTPUT} / {STAGING}')
        STAGING.mkdir(parents=True,exist_ok=False)
    else:
        require(not OUTPUT.exists() and STAGING.is_dir(),
                'Resume requires existing INCOMPLETE Stage6 staging directory')
    C.seed_all(42)
    processor=C.processor_new(a.offline_model)
    device=C.cuda_device()
    try:
        for code,method,variant in MODELS:
            for fold in range(1,6):
                name=f'{method}_{variant}_fold_{fold}'
                folder,summary,run=infos[name]
                fpath=STAGING/fold_filename(method,variant,fold)
                if fpath.exists():
                    load_fold_prediction(fpath,rows,fold)  # Verify before skip
                    print(f'REUSING verified test fold: {name}',flush=True)
                else:
                    print(f'INFER {code} {name}: best_epoch={summary["best_epoch"]}',flush=True)
                    p=inference_fold(rows,method,variant,folder,summary,run,processor,
                                     device,a.offline_model,a.frozen_batch,a.progress_every)
                    C.atomic_csv(fpath,fold_table(rows,p,fold))
                if fold==5:
                    # Verify all five exist, never average incomplete set
                    ps=[load_fold_prediction(STAGING/fold_filename(method,variant,i),rows,i)
                        for i in range(1,6)]
                    en=STAGING/ensemble_filename(method,variant)
                    if en.exists():read_ensemble(en,rows,ps)
                    else:C.atomic_csv(en,ensemble_table(rows,ps))
        save_comparison_and_report(STAGING,rows,lock)
        verify(STAGING,rows,lock)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        os.replace(STAGING,OUTPUT)
        print('FINAL TEST LOCKED AT:',OUTPUT,flush=True)
    except Exception:
        print('FINAL TEST INTERRUPTED; preserve staging and retry with --resume-test:',
              STAGING,file=sys.stderr,flush=True)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true',help='Preflight; no test pixels/inference')
    g.add_argument('--run-test',action='store_true',help='Explicit final test inference')
    g.add_argument('--resume-test',action='store_true',help='Resume interrupted TEST inference')
    g.add_argument('--verify',action='store_true',help='Audit saved ensemble test result')
    ap.add_argument('--offline-model',action='store_true')
    ap.add_argument('--frozen-batch',type=int,default=1)
    ap.add_argument('--progress-every',type=int,default=111)
    a=ap.parse_args()
    require(1<=a.frozen_batch<=4 and a.progress_every>=1,'Unsafe inference batch/progress')
    lock,cfg,rows,infos=preflight()
    if a.check:
        print('PASS_E6_STAGE6_CHECK | no test pixels read | no files written')
    elif a.verify:
        require(OUTPUT.is_dir(),'Final test output does not exist')
        verify(OUTPUT,rows,lock)
    else:run_test(a,lock,rows,infos)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'E6 STAGE 6 FAILED: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise SystemExit(1)
