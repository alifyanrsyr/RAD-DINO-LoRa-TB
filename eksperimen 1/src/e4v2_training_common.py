#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os, random, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
E4_ROOT = ROOT.parent / 'eksperimen4'
SRC = Path(__file__).resolve().parent
CONFIG_PATH = SRC / '02_training_config_e4v2_LOCKED.json'
CONFIG_EXPECTED_SHA = 'a9b9d2eda5e260e20802144363768f3ba18b467b017ff0e5222e08b6504c0b63'
SIZE = 518
MODEL_COMMIT_EXPECTED = '110cbc18d5133582e320b43d53bf5c44e410c936'
KEY = ['source_archive','archive_member_path']
META = KEY + ['source_zip','patient_id','y_true','split']
FOLDS = ROOT/'data/manifests/experiment4_fixedtest257_dev1411_5fold_seed42_v1'
AUDIT = ROOT/'audits/02_dataloader_validation/02_e4v2_dataloader_validation_report.json'
STAGE5 = E4_ROOT/'data/processed/experiment4_letterbox518_v1'
TEST_SOURCE = E4_ROOT/'data/manifests/experiment4_split_70_15_15_seed42.csv'


def need(ok, why):
    if not bool(ok): raise RuntimeError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_json(path):
    need(Path(path).is_file(),f'Missing required JSON: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    need(Path(path).is_file(),f'Missing required CSV: {path}')
    return pd.read_csv(path,dtype=str,keep_default_na=False,low_memory=False)


def atomic_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.writing')
    q.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    os.replace(q,p)


def atomic_csv(path,df):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.writing');df.to_csv(q,index=False);os.replace(q,p)


def atomic_torch(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.writing');torch.save(data,q);os.replace(q,p)


def load_config():
    need(CONFIG_PATH.is_file(),f'Locked E4v2 configuration missing: {CONFIG_PATH}')
    need(sha(CONFIG_PATH)==CONFIG_EXPECTED_SHA,'E4v2 training configuration LOCK changed')
    c=read_json(CONFIG_PATH)
    need(c.get('status')=='LOCKED_PROTOCOL_E4V2_ALIGNED_WITH_E6_2026-10-23', 'E4v2 configuration not locked')
    need(c.get('cross_validation') is True and c.get('folds')==5,'CV protocol changed')
    need(c.get('frozen_head_batch_size')==64 and c.get('lora_micro_batch_size')==1 and
         c.get('lora_gradient_accumulation')==16,'Training batches changed')
    need(c.get('seed')==42 and c.get('max_epochs')==30 and c.get('early_stopping_patience')==5,
         'Locked training protocol changed')
    return c


def safe_png_path(variant, rel):
    need('\\' not in rel and ':' not in rel and isinstance(rel,str),f'Unsafe path: {rel}')
    p=Path(rel)
    need(not p.is_absolute() and '..' not in p.parts and len(p.parts)==2 and
         p.parts[0]==f'{variant}_518' and p.suffix.lower()=='.png',f'Unsafe PNG path: {rel}')
    return STAGE5/p


def preflight(fold, require_dataloader_audit=True):
    need(1 <= fold <= 5, 'Fold must be 1..5')
    cfg=load_config()
    lock_path=FOLDS/'01_e4_cv85_fold_lock_report.json'
    fold_report=read_json(lock_path)
    need(fold_report.get('status')=='PASS_E4_CV85_FOLDS_LOCKED', 'E4v2 5-fold lock not PASS')
    need(sha(lock_path)==cfg['locked_e4v2_fold_report_sha256'],
         'E4v2 fold report SHA256 changed')
    need(sha(FOLDS/'development_1411_5fold_LOCKED.csv')==
         cfg['locked_e4v2_development_manifest_sha256'],
         'E4v2 development manifest SHA256 changed')
    source_files={
       'split':TEST_SOURCE,
       'old_cv':E4_ROOT/'data/manifests/experiment4_train_5fold_seed42.csv',
       'processed':STAGE5/'processed_manifest.csv',
       'audit':E4_ROOT/'data/audit/processed_dataset_final_audit/processed_png_audit_report.json'
    }
    for name,path in source_files.items():
        expect=cfg['expected_hashes'][{
          'split':'e4_split_sha256', 'old_cv':'e4_original_cv_sha256',
          'processed':'e4_processed_manifest_sha256','audit':'e4_audit_sha256'}[name]]
        need(path.is_file() and sha(path)==expect,f'Original E4 source changed: {name}')
    need(fold_report.get('no_patient_leakage') and
         fold_report.get('each_development_image_oof_once') and
         fold_report.get('each_development_patient_oof_once') and
         fold_report.get('same_test_patients_as_original_e4') and
         fold_report.get('test_pixel_accessed') is False,
         'E4v2 patient/fixed test lock not valid')
    for rel,h in fold_report['output_sha256'].items():
        part=Path(rel)
        need(not part.is_absolute() and '..' not in part.parts,'Unsafe fold artifact path')
        q=FOLDS/part
        need(q.is_file() and sha(q)==h,f'E4v2 fold artifact altered: {rel}')
    if require_dataloader_audit:
        dr=read_json(AUDIT)
        need(dr.get('status')=='PASS_E4V2_DATALOADER_VALIDATION' and
             dr.get('checked_combinations')==20 and
             dr.get('test_pixel_accessed') is False and
             dr.get('model_training_performed') is False and
             dr.get('source_fold_report_sha256')==sha(lock_path),
             'E4v2 DataLoader audit incompatible. Run 02_validasi_dataloader_e4v2.py --run')
    dev=read_csv(FOLDS/'development_1411_5fold_LOCKED.csv')
    need(len(dev)==1411 and dev.patient_id.nunique()==1091 and
         dev.groupby('patient_id').cv_fold.nunique().eq(1).all(),
         'E4v2 development size or group integrity mismatch')
    need(dev.e4_original_split.isin(('train','validation')).all() and
         not dev[KEY].duplicated().any() and
         dev.y_true.isin(('0','1')).all(),
         'E4v2 includes unexpected source rows')
    tr=read_csv(FOLDS/f'fold_{fold}/train.csv')
    va=read_csv(FOLDS/f'fold_{fold}/validation.csv')
    need(tr.split.eq('train').all() and va.split.eq('validation').all(),
         'Fold row split labels invalid')
    need(len(tr)+len(va)==1411 and
         tr.patient_id.nunique()+va.patient_id.nunique()==1091,
         'E4v2 fold sizes mismatch')
    need(not set(tr.patient_id).intersection(va.patient_id),
         f'E4v2 patient leakage at fold {fold}')
    need(tr.y_true.nunique()==2 and va.y_true.nunique()==2,
         'Class missing in fold')
    need(tr.cv_fold.ne(str(fold)).all() and va.cv_fold.eq(str(fold)).all(),
         'Incorrect fold roles')
    identities=lambda x:set(map(tuple,x[KEY].to_numpy()))
    need(not identities(tr).intersection(identities(va)) and
         identities(tr).union(identities(va))==identities(dev),
         'Incomplete or overlapping fold identities')
    for role,df in [('train',tr),('validation',va)]:
        need(not df[KEY].duplicated().any(),f'Duplicate {role} keys')
        for variant in ['raw','clahe']:
            need(df[f'{variant}_path'].nunique()==len(df),
                 f'Duplicate {role} {variant} PNG paths')
            for rel in df[f'{variant}_path']:
                need(safe_png_path(variant,rel).is_file(),f'Missing PNG {rel}')
    fp={
       'training_config':sha(CONFIG_PATH),
       'stage1_e4v2_fold_report':sha(lock_path),
       'e4v2_dev_locked':sha(FOLDS/'development_1411_5fold_LOCKED.csv'),
       'e4_original_split_sha256':sha(TEST_SOURCE),
       'e4_processed_manifest_sha256':sha(STAGE5/'processed_manifest.csv'),
       'fold_train_manifest':sha(FOLDS/f'fold_{fold}/train.csv'),
       'fold_validation_manifest':sha(FOLDS/f'fold_{fold}/validation.csv'),
       'fold':fold, 'train_images':len(tr), 'validation_images':len(va)
    }
    if require_dataloader_audit:
        fp['stage2_dataloader_sha256']=sha(AUDIT)
    print(f'E4V2 LOCK PASS | fold={fold} train={len(tr)} validation={len(va)} | '
          'original E4 test PNGs untouched',flush=True)
    return cfg, {'train':tr.sort_values(KEY).reset_index(drop=True),
                 'validation':va.sort_values(KEY).reset_index(drop=True)}, fp


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


class PNGDataset(Dataset):
    def __init__(self,rows,variant,processor):
        need(variant in ('raw','clahe'),'Unknown variant')
        need(len(rows)>0 and rows.split.isin(['train','validation']).all(),
             'Pixel access is limited to train/validation')
        self.rows=rows.reset_index(drop=True); self.variant=variant;self.processor=processor
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows.iloc[index];f=safe_png_path(self.variant,r[self.variant+'_path'])
        with Image.open(f) as im:
            need(im.mode=='L' and im.size==(518,518),f'Unexpected PNG encoding: {f}')
            rgb=im.convert('RGB')
            x=self.processor(images=rgb, do_resize=False, do_center_crop=False,
                             return_tensors='pt')['pixel_values'].squeeze(0)
        need(tuple(x.shape)==(3,518,518) and torch.isfinite(x).all().item(),
             f'Bad image tensor {f}')
        y=int(r['y_true']);need(y in (0,1),'Bad class label')
        key=r['source_archive']+'::'+r['archive_member_path']
        return x,torch.tensor(y,dtype=torch.long),key,str(r['patient_id'])


def processor_new(offline=False):
    from transformers import AutoImageProcessor
    p=AutoImageProcessor.from_pretrained('microsoft/rad-dino',revision=MODEL_COMMIT_EXPECTED,local_files_only=offline)
    need(p.__class__.__name__=='BitImageProcessor' and p.do_rescale and
         p.do_normalize and p.do_convert_rgb, 'Unexpected RAD-DINO processor')
    need(np.allclose(p.image_mean,[.5307]*3,atol=1e-5,rtol=0) and
         np.allclose(p.image_std,[.2583]*3,atol=1e-5,rtol=0) and
         math.isclose(float(p.rescale_factor),1/255,rel_tol=1e-6),
         'Processor normalization not locked to E4')
    return p


def backbone_new(offline=False):
    from transformers import AutoModel
    m=AutoModel.from_pretrained('microsoft/rad-dino',revision=MODEL_COMMIT_EXPECTED,local_files_only=offline)
    need(getattr(m.config,'_commit_hash',MODEL_COMMIT_EXPECTED)==MODEL_COMMIT_EXPECTED,
         'RAD-DINO commit differs from locked checkpoint')
    need(int(m.config.hidden_size)==768,'Unexpected RAD-DINO feature size')
    return m


def lora_wrap(backbone):
    from peft import LoraConfig,get_peft_model
    cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                   target_modules=['query','value'],layers_to_transform=[8,9,10,11],
                   layers_pattern='layer',bias='none')
    m=get_peft_model(backbone,cfg)
    trained=[(name,p) for name,p in m.named_parameters() if p.requires_grad]
    need(trained and all('lora_A' in name or 'lora_B' in name for name,_ in trained),
         'Unexpected backbone params selected for fine-tuning')
    need(sum(p.numel() for _,p in trained)==98304,
         'LoRA parameter count is not 98,304. Verify PEFT architecture.')
    return m


def dataset_loader(rows,variant,processor,batch_size,shuffle=False,epoch=None):
    generator=torch.Generator()
    generator.manual_seed(42+(epoch or 0))
    return DataLoader(PNGDataset(rows,variant,processor),batch_size=batch_size,
                      shuffle=shuffle,generator=generator,num_workers=0,
                      pin_memory=torch.cuda.is_available(),drop_last=False)


def classification_metrics(y,probs):
    y=np.asarray(y,dtype=int); p=np.asarray(probs,dtype=float)
    need(len(y)==len(p) and len(y)>0 and set(y.tolist())=={0,1} and
         np.isfinite(p).all(), 'Invalid validation metrics input')
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return {'auroc':float(roc_auc_score(y,p)),
            'accuracy':float((pred==y).mean()),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'sensitivity_dr':float(tp/(tp+fn)),
            'specificity_ds':float(tn/(tn+fp)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'threshold':.5}


def version_info():
    import transformers
    import sklearn
    v={'torch':torch.__version__,'transformers':transformers.__version__,
       'sklearn':sklearn.__version__,'python':sys.version.split()[0]}
    try:
        import peft
        v['peft']=peft.__version__
    except ImportError:
        v['peft']=None
    return v


def cuda_device():
    need(torch.cuda.is_available(), 'CUDA GPU required; training on CPU is intentionally disabled')
    d=torch.device('cuda:0')
    return d


def check_smoke(method,variant,fingerprints,gradient_checkpointing=False):
    p=ROOT/'audits'/'03_gpu_smoke'/f'{method}_{variant}.json'
    d=read_json(p)
    need(d.get('status')=='PASS_E4V2_GPU_SMOKE' and
         d.get('method')==method and d.get('variant')==variant and
         d.get('config_sha256')==sha(CONFIG_PATH) and
         d.get('source_processed_sha256')==sha(STAGE5/'processed_manifest.csv') and
         d.get('gradient_checkpointing')==gradient_checkpointing,
         f'Run matching GPU smoke test before training: {p}')
    return d


def record_smoke(method,variant,gradient_checkpointing=False,extra=None):
    p=ROOT/'audits'/'03_gpu_smoke'/f'{method}_{variant}.json'
    obj={'status':'PASS_E4V2_GPU_SMOKE','method':method,'variant':variant,
         'config_sha256':sha(CONFIG_PATH),
         'source_processed_sha256':sha(STAGE5/'processed_manifest.csv'),
         'gradient_checkpointing':gradient_checkpointing,
         'test_pixel_accessed':False,
         'detail':extra or {}}
    atomic_json(p,obj)
    return p


def validation_prediction_csv(rows,probs):
    p=np.asarray(probs,dtype=float)
    need(len(p)==len(rows) and np.isfinite(p).all(),'Prediction array count mismatch')
    out=rows[KEY+['patient_id','y_true','split','cv_fold']].copy()
    out['p_dr']=p
    out['pred_0_5']=(p>=.5).astype(int)
    need(out['split'].eq('validation').all(),'Attempt to save nonvalidation probabilities')
    return out


def rng_state():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def torch_load_trusted(path, map_location='cpu'):
    # These .pt files are written by *this* script in the local machine.
    # Never use weights_only=False on untrusted external checkpoints.
    return torch.load(path,map_location=map_location,weights_only=False)


def save_result_manifest(output,method,variant,config,fingerprints,best_epoch,best_score,epoch,history,model_commit=None):
    atomic_json(Path(output)/'run_summary.json',{
        'status':'PASS_E4V2_FOLD_TRAINING_COMPLETE','experiment':'experiment4v2',
        'method':method,'variant':variant,'model_id':'microsoft/rad-dino',
        'model_commit':model_commit,'best_epoch':int(best_epoch),
        'best_validation_auroc':float(best_score),'last_epoch':int(epoch),
        'epochs_recorded':len(history),'checkpoints_selected_on':'validation_auroc',
        'test_pixel_read':False,'test_inference':False,'five_fold_cv':True,
        'fold':int(Path(output).name.split('_fold_')[-1]),
        'fingerprints':fingerprints,'versions':version_info(),
        'validation_images': int(fingerprints['validation_images']),
        'train_images': int(fingerprints['train_images']),
        'frozen_head_batch_size':config['frozen_head_batch_size'],
        'lora_micro_batch_size':config['lora_micro_batch_size'],
        'lora_gradient_accumulation':config['lora_gradient_accumulation']})
