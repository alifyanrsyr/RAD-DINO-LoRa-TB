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

E7_ROOT = Path(__file__).resolve().parents[1]
E7_SRC = E7_ROOT / 'src'
E6_SRC = E7_ROOT.parent / 'eksperimen6' / 'src'
if not (E6_SRC / 'e6_training_common.py').is_file():
    raise RuntimeError(f'Tidak ada utility E6 (jangan pindah folder E6): {E6_SRC}')
sys.path.insert(0, str(E6_SRC))
import e6_training_common as C

MODELS = [('M1','frozen','raw'),('M2','frozen','clahe'),
          ('M3','lora','raw'),('M4','lora','clahe')]
KEY = ['source_archive','archive_member_path']
ID = KEY + ['patient_id','y_true']
CONFIG = E7_SRC/'07_training_config_e7_LOCKED.json'
CONFIG_SHA = 'cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'
OOF = E7_ROOT/'output/cross_validation/08_e7_oof_model_selection_v1'
OOF_LOCK = OOF/'08_e7_oof_model_selection_LOCKED.json'
OOF_SHA = 'f46dd75163c6232d630f4a76f2fc9d786210f24a836373b76c8ae4fe36509787'
THRESHOLD_DIR = E7_ROOT/'output/threshold/11_e7_threshold_oof_review_v1'
THRESHOLD_REPORT = THRESHOLD_DIR/'11_e7_threshold_report.json'
TRAINING = E7_ROOT/'output/training'
OUTPUT = E7_ROOT/'output/testing/12_e7_five_fold_ensemble_holdout_v1'
STAGING = OUTPUT.with_name(OUTPUT.name+'_staging')
REPORT_FILE = '12_e7_ensemble_holdout_test_LOCKED.json'
COMPARE_FILE = '12_e7_ensemble_test_comparison.csv'
EXPLORATORY_FILE = '12_e7_m4_threshold_044_EXPLORATORY.csv'
STATUS = 'PASS_E7_REUSED_HOLDOUT_5FOLD_ENSEMBLE'
BASELINE = 0.50
M4_ALT = 0.44  # eksploratif, dikunci dari OOF SEBELUM testing


def need(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def jread(p):
    return C.read_json(p)


def digest(p):
    return C.sha(p)


def run_name(method, variant, fold):
    return f'{method}_{variant}_fold_{fold}'


def fold_filename(method, variant, fold):
    return f'12_e7_test_predictions_{method}_{variant}_fold_{fold}.csv'


def ensemble_filename(method, variant):
    return f'12_e7_test_ensemble_predictions_{method}_{variant}.csv'


def identity_keys(rows):
    return (rows.source_archive+'::'+rows.archive_member_path).tolist()


def metrics_at_threshold(labels, probs, t=0.5):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
    y=np.asarray(labels, dtype=int)
    p=np.asarray(probs, dtype=float)
    need(len(y)==333 and len(p)==333 and set(np.unique(y))=={0,1} and
         np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Metrik test tidak valid')
    pred=(p>=float(t)).astype(int)
    tn,fp,fn,tp=[int(v) for v in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    need(tn+fp==166 and fn+tp==167, 'Jumlah label DS/DR test salah')
    return {'auroc':float(roc_auc_score(y,p)),
            'accuracy':float(np.mean(pred==y)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'sensitivity_dr':float(tp/(tp+fn)),
            'specificity_ds':float(tn/(tn+fp)),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp,'threshold':float(t)}


def metadata_only_test(cfg):
    """Boleh membaca label+path manifest; TIDAK membuka PNG test."""
    need(digest(C.TEST_SOURCE)==cfg['reference_e5_test_manifest_sha256'],
         'Manifest E5 TEST berubah')
    # Base utility for original E6 protocol, reused READ-ONLY; no test pixels.
    C.load_config()
    _cfg,_,_evidence = C.preflight(1)
    need(digest(C.STAGE5/'processed_manifest.csv')==
         _cfg['expected_hashes']['e5_processed_manifest_sha256'],
         'Processed PNG manifest berubah')
    test=C.read_csv(C.TEST_SOURCE)
    man=C.read_csv(C.STAGE5/'processed_manifest.csv')
    need(set(ID+['split','raw_path','clahe_path']).issubset(man.columns),
         'Kolom processed manifest tidak lengkap')
    rows=man.loc[man.split.eq('test')].copy().sort_values(KEY).reset_index(drop=True)
    expect=test.sort_values(KEY).reset_index(drop=True)
    need(len(rows)==333 and len(expect)==333 and expect.split.eq('test').all(),
         'Holdout tidak 333 citra')
    need(not rows[KEY].duplicated().any() and not expect[KEY].duplicated().any(),
         'Duplikat identitas test')
    need(rows[ID].equals(expect[ID]),'Identitas/label processed test != E5 test.csv')
    need(rows.patient_id.nunique()==259 and
         rows.y_true.value_counts().to_dict()=={'0':166,'1':167} and
         rows.groupby('patient_id').y_true.nunique().eq(1).all(),
         'Distribusi 259 pasien/166 DS/167 DR tidak cocok')
    dev=C.read_csv(C.FOLDS/'e6_development_1887_5fold_LOCKED.csv')
    need(len(dev)==1887 and dev.patient_id.nunique()==1471 and
         not set(dev.patient_id).intersection(rows.patient_id) and
         not set(map(tuple,dev[KEY].to_numpy())).intersection(
             set(map(tuple,rows[KEY].to_numpy()))),
         'Kebocoran pasien/citra development ke test')
    for variant in ('raw','clahe'):
        need(rows[f'{variant}_path'].nunique()==333,
             f'Path PNG {variant} duplikat')
        for rel in rows[f'{variant}_path']:
            need(C.safe_png_path(variant,rel).is_file(),f'File PNG test hilang: {rel}')
    return rows


def safe_snapshot(base, relative):
    rel=Path(str(relative))
    need(not rel.is_absolute() and '..' not in rel.parts and
         len(rel.parts)==2 and rel.parts[0]=='best_candidates' and
         rel.parts[1].startswith('epoch_'),f'Path best LoRA tidak aman: {relative}')
    target=base/rel
    need(target.is_dir(),f'Checkpoint LoRA hilang: {target}')
    return target


def preflight():
    need(CONFIG.is_file() and digest(CONFIG)==CONFIG_SHA,
         'Training config E7 hilang atau tidak cocok dengan SHA terkunci')
    cfg=jread(CONFIG)
    need(cfg['status']=='LOCKED_E7_FOUR_MODELS' and
         cfg['backbone_revision']==C.MODEL_COMMIT_EXPECTED and
         cfg['classification_head_dropout']==0.3 and
         cfg['decision_threshold_baseline']==BASELINE and
         cfg['n_training_runs']==20 and
         [(v['model'],v['method'],v['variant']) for v in cfg['model_variants']]==MODELS,
         'Konfigurasi E7 tidak cocok dengan protokol')
    need(OOF_LOCK.is_file() and digest(OOF_LOCK)==OOF_SHA,
         'OOF LOCKED E7 berubah atau tidak ada')
    lock=jread(OOF_LOCK)
    need(lock.get('status')=='PASS_E7_OOF_MODEL_SELECTION_LOCKED' and
         lock.get('primary_model')=='lora_clahe' and
         lock.get('primary_selected_before_e7_test') is True and
         abs(float(lock['primary_model_oof_auroc'])-0.7335417527904092)<1e-12 and
         lock.get('n_training_runs')==20 and lock.get('n_models')==4 and
         lock.get('n_development_images')==1887 and
         lock.get('n_test_images_REFERENCE_ONLY')==333 and
         lock.get('test_pixel_read') is False and
         lock.get('test_inference_performed') is False,
         'Status OOF E7 belum sah untuk TEST')
    need(lock['source_sha256']['e7_training_config']==CONFIG_SHA and
         lock['source_sha256']['e6_development_manifest']==
         cfg['reference_e6_fold_manifest_sha256'] and
         lock['source_sha256']['e5_fixed_holdout_manifest_reference_ONLY']==
         cfg['reference_e5_test_manifest_sha256'],
         'Provenance OOF config/fold/test berubah')
    need(digest(C.FOLDS/'e6_development_1887_5fold_LOCKED.csv')==
         lock['source_sha256']['e6_development_manifest'] and
         digest(C.FOLDS/'02_e6_5fold_lock_report.json')==
         lock['source_sha256']['e6_fold_lock'],
         'Referensi fold E6 berubah')
    # Verify immutable OOF artifacts
    for name,sha in lock['output_sha256'].items():
        need(digest(OOF/name)==sha,f'Artefak OOF telah berubah: {name}')
    need([r['model'] for r in lock['models']][0]=='lora_clahe',
         'Model utama tidak ranked paling tinggi dalam OOF')
    # Threshold exploratory report is used ONLY for predetermined optional 0.44.
    need(THRESHOLD_REPORT.is_file(),
         f'Laporan threshold OOF hilang: {THRESHOLD_REPORT}')
    thr=jread(THRESHOLD_REPORT)
    need(thr.get('status')=='PASS_E7_EXPLORATORY_OOF_THRESHOLD_REVIEW' and
         thr.get('primary_model_from_locked_oof')=='lora_clahe' and
         thr.get('baseline_threshold_unchanged')==BASELINE and
         thr.get('no_threshold_adopted_for_holdout') is True and
         thr.get('no_test_data_read') is True and
         thr['source_sha256']['oof_locked_json']==OOF_SHA and
         abs(float(thr['full_oof_balanced_accuracy_best_candidate_EXPLORATORY_NOT_LOCKED']['threshold'])-M4_ALT)<1e-10,
         'Eksplorasi threshold M4/0.44 tidak didukung laporan OOF')
    for name,sha in thr['output_sha256'].items():
        need(digest(THRESHOLD_DIR/name)==sha,f'File threshold berubah: {name}')

    run_info={}
    for code,method,variant in MODELS:
        for fold in range(1,6):
            name=run_name(method,variant,fold)
            folder=TRAINING/name
            provenance=lock['run_artifact_sha256'].get(name)
            need(isinstance(provenance,dict) and provenance,
                 f'Provenance checkpoint tidak ditemukan: {name}')
            for relative,sha in provenance.items():
                need(digest(folder/relative)==sha,
                     f'Hash training/checkpoint berubah: {name}/{relative}')
            run=jread(folder/'run_config.json')
            summary=jread(folder/'run_summary.json')
            need(summary.get('status')=='PASS_E7_FOLD_TRAINING_COMPLETE' and
                 summary.get('model')==f'{method}_{variant}' and
                 summary.get('fold')==fold and
                 summary.get('test_read') is False and
                 summary.get('locked_config_sha256')==CONFIG_SHA and
                 summary.get('model_commit')==cfg['backbone_revision'] and
                 1<=int(summary['best_epoch'])<=int(summary['last_epoch'])<=30 and
                 run.get('experiment')=='experiment7' and
                 (run.get('config_sha256') if method=='frozen' else
                  run.get('locked_config_sha256'))==CONFIG_SHA,
                 f'Ringkasan/model training tidak sesuai E7: {name}')
            need(summary.get('source_e6_fingerprints')==
                 (run.get('fold_source_fingerprints') if method=='frozen' else
                  run.get('source_e6_fingerprints')) and
                 summary['source_e6_fingerprints']['fold']==fold,
                 f'Fingerprint train/val tidak cocok: {name}')
            if method=='lora':
                pointer=jread(folder/'best_checkpoint.json')
                need(pointer['epoch']==summary['best_epoch'] and
                     abs(float(pointer['validation_auroc'])-
                         float(summary['best_validation_auroc']))<1e-9 and
                     summary['best_snapshot_relative_path']==pointer['snapshot_relative_path'],
                     f'Pointer checkpoint LoRA tidak cocok: {name}')
                best=safe_snapshot(folder,pointer['snapshot_relative_path'])
                conf=jread(best/'adapter_config.json')
                need(int(conf['r'])==8 and int(conf['lora_alpha'])==8 and
                     set(conf['target_modules'])=={'query','value'} and
                     list(conf['layers_to_transform'])==[8,9,10,11],
                     f'Konfigurasi adapter LoRA berubah: {name}')
                need((best/'adapter_model.safetensors').is_file(),
                     f'Adapter LoRA hilang: {name}')
            else:
                best=folder
            need((best/'best_classifier.pt').is_file(),
                 f'Classifier terbaik tidak ditemukan: {name}')
            run_info[name]=(folder,best,summary,run)
    rows=metadata_only_test(cfg)
    print('PASS_E7_TEST_PREFLIGHT: OOF lock+threshold+20 best checkpoints verified',flush=True)
    print('333 test images / 259 patients / DS=166 / DR=167; 0 test pixels opened',flush=True)
    print('M4 LoRA CLAHE LOCKED before test | PRIMARY threshold 0.50 | exploratory M4=0.44',flush=True)
    return cfg,lock,rows,run_info


class TestPNG(Dataset):
    def __init__(self, rows, variant, processor):
        need(len(rows)==333 and variant in ('raw','clahe') and
             rows.split.eq('test').all(),'Hanya test PNG yang diperbolehkan untuk inferensi')
        self.rows=rows.reset_index(drop=True)
        self.variant=variant
        self.processor=processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row=self.rows.iloc[index]
        path=C.safe_png_path(self.variant,row[self.variant+'_path'])
        with Image.open(path) as im:
            need(im.mode=='L' and im.size==(518,518),f'Test PNG salah: {path}')
            x=self.processor(images=im.convert('RGB'),do_resize=False,
                             do_center_crop=False,return_tensors='pt')['pixel_values'].squeeze(0)
        need(tuple(x.shape)==(3,518,518) and torch.isfinite(x).all().item(),
             'Tensor input test tidak valid')
        return (x,torch.tensor(int(row.y_true),dtype=torch.long),
                str(row.patient_id),row.source_archive+'::'+row.archive_member_path)


def load_head(best,method,summary,device):
    # Trusted: only files with SHA-256 verified against immutable OOF lock.
    obj=C.torch_load_trusted(best/'best_classifier.pt')
    need(obj.get('best_epoch')==summary['best_epoch'] and
         obj.get('model_commit')==summary['model_commit'] and
         obj.get('feature')=='CLS' and
         obj.get('head')=='Dropout(0.3)+Linear(768,2)',
         'Checkpoint classifier tidak sejalan dengan protokol')
    if method=='lora':
        need(obj.get('fingerprints')==summary['source_e6_fingerprints'],
             'Fingerprint classifier LoRA berubah')
    else:
        need(obj.get('source_fingerprints')==summary['source_e6_fingerprints'],
             'Fingerprint classifier Frozen berubah')
    head=nn.Sequential(nn.Dropout(p=0.3),nn.Linear(768,2))
    head.load_state_dict(obj['classifier_state_dict'],strict=True)
    return head.to(device).eval()


def infer_one_fold(rows,method,variant,folder,best,summary,processor,device,offline,batch,progress):
    backbone=C.backbone_new(offline)
    need(getattr(backbone.config,'_commit_hash',C.MODEL_COMMIT_EXPECTED)==summary['model_commit'],
         'Backbone pretrained revision berbeda')
    if method=='lora':
        from peft import PeftModel
        backbone=PeftModel.from_pretrained(backbone,str(best),is_trainable=False)
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone=backbone.to(device).eval()
    head=load_head(best,method,summary,device)
    loader=DataLoader(TestPNG(rows,variant,processor),batch_size=batch if method=='frozen' else 1,
                      shuffle=False,drop_last=False,num_workers=0,pin_memory=True)
    yvals=[];keys=[];probs=[];seen=0
    try:
        with torch.inference_mode():
            for x,y,_patients,ids in loader:
                with torch.cuda.amp.autocast(enabled=(method=='lora')):
                    feat=backbone(pixel_values=x.to(device,non_blocking=True)).last_hidden_state[:,0,:]
                    logits=head(feat).float()
                need(tuple(logits.shape)==(len(y),2) and torch.isfinite(logits).all().item(),
                     'Output logits tidak valid')
                p=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
                need(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),
                     'Probabilitas test tidak valid')
                probs.extend(p.tolist());yvals.extend(y.tolist());keys.extend(ids)
                seen+=len(y)
                if seen%progress==0 or seen==len(rows):
                    print(f'   {method}_{variant}: {seen}/333 citra',flush=True)
    finally:
        del backbone,head,loader
        gc.collect()
        torch.cuda.empty_cache()
    need(seen==333 and keys==identity_keys(rows) and
         yvals==rows.y_true.astype(int).tolist(),
         'Urutan identitas atau label test tidak cocok')
    return np.asarray(probs,dtype=float)


def fold_table(rows,fold,probs):
    df=rows[ID].copy()
    df['fold']=fold
    df['p_dr']=np.asarray(probs,dtype=float)
    return df


def read_fold(path,rows,fold):
    df=C.read_csv(path)
    need(list(df.columns)==ID+['fold','p_dr'] and len(df)==333 and
         df[ID].equals(rows[ID]) and df.fold.eq(str(fold)).all(),
         f'Prediksi fold rusak atau tidak cocok: {path}')
    p=np.asarray(pd.to_numeric(df.p_dr,errors='raise'),dtype=float)
    need(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),
         f'Prediksi fold di luar [0,1]: {path}')
    return p


def ensemble_table(rows,ps):
    need(len(ps)==5 and all(len(p)==333 for p in ps),'Ensemble butuh tepat 5 fold')
    p=np.mean(np.stack(ps,axis=0),axis=0)
    df=rows[ID].copy()
    df['p_dr_mean_5fold']=p
    df['pred_0_5']=(p>=BASELINE).astype(int)
    df['correct']=(df.pred_0_5.to_numpy()==rows.y_true.astype(int).to_numpy()).astype(int)
    return df


def read_ensemble(path,rows,ps):
    df=C.read_csv(path)
    need(list(df.columns)==ID+['p_dr_mean_5fold','pred_0_5','correct'] and
         len(df)==333 and df[ID].equals(rows[ID]),
         f'File ensemble tidak sesuai: {path}')
    exp=ensemble_table(rows,ps)
    p=pd.to_numeric(df.p_dr_mean_5fold,errors='raise').to_numpy(dtype=float)
    need(np.allclose(p,exp.p_dr_mean_5fold.to_numpy(),rtol=0,atol=1e-14) and
         df.pred_0_5.astype(int).tolist()==exp.pred_0_5.tolist() and
         df.correct.astype(int).tolist()==exp.correct.tolist(),
         f'Ensemble bukan rata-rata tepat lima fold: {path}')
    return p


def collected_metrics(folder,rows):
    results=[];hashes={};m4probs=None
    for code,method,variant in MODELS:
        ps=[]
        for fold in range(1,6):
            filename=fold_filename(method,variant,fold)
            path=folder/filename
            ps.append(read_fold(path,rows,fold))
            hashes[filename]=digest(path)
        filename=ensemble_filename(method,variant)
        path=folder/filename
        probs=read_ensemble(path,rows,ps)
        hashes[filename]=digest(path)
        scores=metrics_at_threshold(rows.y_true.astype(int).to_numpy(),probs,BASELINE)
        results.append({'model_code':code,'model':f'{method}_{variant}',
                        'method':method,'variant':variant,
                        'primary_preselected':code=='M4','n_folds':5,
                        'n_test_images':333,'n_test_patients':259,**scores})
        print(f' {code} ({method}_{variant}): AUROC={scores["auroc"]:.6f} '
              f'MacroF1={scores["macro_f1"]:.6f} '
              f'BalAcc={scores["balanced_accuracy"]:.6f} '
              f'SensDR={scores["sensitivity_dr"]:.6f} '
              f'SpecDS={scores["specificity_ds"]:.6f}',flush=True)
        if code=='M4':m4probs=probs
    extra=metrics_at_threshold(rows.y_true.astype(int).to_numpy(),m4probs,M4_ALT)
    return pd.DataFrame(results),extra,hashes


def save_report(folder,rows,lock):
    cmp,extra,hashes=collected_metrics(folder,rows)
    C.atomic_csv(folder/COMPARE_FILE,cmp)
    hashes[COMPARE_FILE]=digest(folder/COMPARE_FILE)
    alt={'model':'lora_clahe','model_code':'M4',
         'analysis':'EXPLORATORY_ONLY_NOT_PRIMARY',
         'threshold_selected_from':'E7 development OOF before any E7 test inference',
         'n_test_images':333,**extra}
    C.atomic_csv(folder/EXPLORATORY_FILE,pd.DataFrame([alt]))
    hashes[EXPLORATORY_FILE]=digest(folder/EXPLORATORY_FILE)
    primary=cmp.loc[cmp.model_code=='M4'].iloc[0].to_dict()
    report={
        'status':STATUS,'experiment':'experiment7',
        'evaluation_protocol':'fixed E5 333-image holdout; five independent best-fold checkpoints averaged per model',
        'holdout_reused_from_e5_e6':True,
        'test_result_interpretation':'EXPLORATORY_REUSED_HOLDOUT_NOT_NEW_INDEPENDENT_TEST',
        'primary_model':'lora_clahe','primary_model_code':'M4',
        'primary_selection_criterion':'pooled image-level OOF AUROC',
        'primary_selection_locked_before_testing':True,
        'primary_model_oof_auroc':lock['primary_model_oof_auroc'],
        'primary_model_test_metrics':primary,
        'primary_threshold':BASELINE,
        'exploratory_M4_threshold':M4_ALT,
        'exploratory_M4_test_metrics':alt,
        'test_based_model_selection':False,
        'test_based_threshold_tuning':False,
        'test_cohort':{'images':333,'patients':259,'DS-TB':166,'DR-TB':167},
        'model_comparison':cmp.to_dict(orient='records'),
        'test_manifest_sha256':digest(C.TEST_SOURCE),
        'e7_config_sha256':digest(CONFIG),
        'oof_model_selection_lock_sha256':digest(OOF_LOCK),
        'oof_threshold_report_sha256':digest(THRESHOLD_REPORT),
        'checkpoint_source_sha256':lock['run_artifact_sha256'],
        'training_during_test':False,
        'inference':{'aggregation':'arithmetic mean of five best fold probabilities',
                     'checkpoints_per_configuration':5,
                     'prediction_unit':'image',
                     'positive_label':'DR-TB (1)',
                     'fold_checkpoint_selection':'per-fold validation AUROC',
                     'primary_decision_threshold':BASELINE,
                     'secondary_threshold_M4_exploratory':M4_ALT},
        'output_sha256':hashes,
    }
    C.atomic_json(folder/REPORT_FILE,report)
    seal=folder/(REPORT_FILE+'.sha256')
    seal.write_text(digest(folder/REPORT_FILE)+'  '+REPORT_FILE+'\n',encoding='utf-8')


def verify(folder,rows,lock):
    report=jread(folder/REPORT_FILE)
    need(report.get('status')==STATUS and
         report.get('primary_model')==lock['primary_model'] and
         report.get('primary_threshold')==BASELINE and
         report.get('exploratory_M4_threshold')==M4_ALT and
         report.get('oof_model_selection_lock_sha256')==digest(OOF_LOCK) and
         report.get('oof_threshold_report_sha256')==digest(THRESHOLD_REPORT) and
         report.get('test_manifest_sha256')==digest(C.TEST_SOURCE),
         'Provenance laporan final test tidak cocok')
    need((folder/(REPORT_FILE+'.sha256')).read_text(encoding='utf-8').strip()==
         digest(folder/REPORT_FILE)+'  '+REPORT_FILE,
         'SHA256 report seal tidak valid')
    cmp,extra,hashes=collected_metrics(folder,rows)
    for file,sha in report['output_sha256'].items():
        need(digest(folder/file)==sha,f'Output test telah dimodifikasi: {file}')
    need(set(hashes).issubset(report['output_sha256']) and
         cmp.to_dict(orient='records')==report['model_comparison'],
         'Metrik dalam laporan tidak cocok dengan prediksi ensemble')
    stored=C.read_csv(folder/COMPARE_FILE)
    for _,row in cmp.iterrows():
        ref=stored.loc[stored.model==row['model']].iloc[0]
        for field in ('auroc','accuracy','macro_f1','balanced_accuracy','sensitivity_dr',
                      'specificity_ds','threshold'):
            need(abs(float(row[field])-float(ref[field]))<1e-12,
                 f'Metrik CSV akhir berubah: {row["model"]}/{field}')
    exp_alt=report['exploratory_M4_test_metrics']
    for field in ('auroc','accuracy','macro_f1','balanced_accuracy','sensitivity_dr',
                  'specificity_ds','threshold'):
        need(abs(float(extra[field])-float(exp_alt[field]))<1e-12,
             f'Threshold alternatif tidak cocok pada metrik {field}')
    for field in ('tn','fp','fn','tp'):
        need(int(extra[field])==int(exp_alt[field]),
             'Confusion matrix threshold alternatif berubah')
    print('PASS_E7_TEST_VERIFIED | M4 primary unchanged | no model inference',flush=True)


def run_test(args,lock,rows,run_info):
    need(torch.cuda.is_available(),'Diperlukan CUDA GPU untuk inferensi test')
    if args.run_test:
        need(not OUTPUT.exists() and not STAGING.exists(),
             f'Output/staging lama terdeteksi; tidak akan overwrite: {OUTPUT} / {STAGING}')
        STAGING.mkdir(parents=True,exist_ok=False)
    else:
        need(not OUTPUT.exists() and STAGING.is_dir(),
             'Resume hanya jika output belum selesai dan staging tersedia')
    C.seed_all(42)
    processor=C.processor_new(args.offline_model)
    device=C.cuda_device()
    try:
        for code,method,variant in MODELS:
            for fold in range(1,6):
                name=run_name(method,variant,fold)
                folder,best,summary,_run=run_info[name]
                file=STAGING/fold_filename(method,variant,fold)
                if file.is_file():
                    read_fold(file,rows,fold)
                    print(f'REUSE {name}: prediksi test sebelumnya valid',flush=True)
                else:
                    print(f'INFER {code} {name} | best epoch={summary["best_epoch"]}',flush=True)
                    probs=infer_one_fold(rows,method,variant,folder,best,summary,
                                         processor,device,args.offline_model,
                                         args.frozen_batch,args.progress_every)
                    C.atomic_csv(file,fold_table(rows,fold,probs))
            ps=[read_fold(STAGING/fold_filename(method,variant,i),rows,i) for i in range(1,6)]
            en=STAGING/ensemble_filename(method,variant)
            if en.exists():read_ensemble(en,rows,ps)
            else:C.atomic_csv(en,ensemble_table(rows,ps))
        save_report(STAGING,rows,lock)
        verify(STAGING,rows,lock)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        os.replace(STAGING,OUTPUT)
        print(f'FINAL TESTING E7 LOCKED: {OUTPUT}',flush=True)
    except Exception:
        print('TERHENTI: staging aman dipertahankan. Gunakan --resume-test. Lokasi:',
              STAGING,file=sys.stderr,flush=True)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Preflight TANPA membaca pixel testing')
    mode.add_argument('--run-test',action='store_true',help='Inferensi test eksplisit; memerlukan CUDA')
    mode.add_argument('--resume-test',action='store_true',help='Lanjutkan inferensi testing yang terputus')
    mode.add_argument('--verify',action='store_true',help='Verifikasi hasil tanpa inferensi baru')
    ap.add_argument('--offline-model',action='store_true',help='Gunakan checkpoint HF lokal saja')
    ap.add_argument('--frozen-batch',type=int,default=1,help='Batch backbone Frozen (1..4)')
    ap.add_argument('--progress-every',type=int,default=111,help='Tampilkan progres per citra')
    args=ap.parse_args()
    need(1<=args.frozen_batch<=4 and args.progress_every>=1,
         'Argumen batch/progres tidak valid')
    _cfg,lock,rows,info=preflight()
    if args.check:
        print('PASS_E7_STAGE12_CHECK | no test pixel access | no GPU inference')
        return
    if args.verify:
        need(OUTPUT.is_dir(),'Hasil testing final belum ditemukan')
        verify(OUTPUT,rows,lock)
        return
    run_test(args,lock,rows,info)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'FAIL_E7_STAGE12 | {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
