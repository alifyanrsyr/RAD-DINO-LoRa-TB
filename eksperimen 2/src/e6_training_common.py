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
E5_ROOT = ROOT.parent / 'eksperimen5'
SRC = Path(__file__).resolve().parent
CONFIG_PATH = SRC / '04_training_config_e6_LOCKED.json'
CONFIG_EXPECTED_SHA = '30c609b8762722da0fc196584d8dfcf9fa3efd3d7dbe07afacdc791e25f4d564'
SIZE = 518
MODEL_COMMIT_EXPECTED = '110cbc18d5133582e320b43d53bf5c44e410c936'
KEY = ['source_archive','archive_member_path']
META = KEY + ['source_zip','patient_id','y_true','split']
FOLDS = ROOT/'data/manifests/experiment6_85dev_15test_5fold_seed42_v1'
AUDIT = ROOT/'audits/03_dataloader_validation/03_e6_dataloader_validation_report.json'
STAGE5 = E5_ROOT/'data/processed/experiment5_letterbox518_v1'
TEST_SOURCE = E5_ROOT/'data/manifests/experiment5_split_70_15_15_seed42_v1/test.csv'


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
    need(CONFIG_PATH.is_file(),f'Locked E6 configuration missing: {CONFIG_PATH}')
    need(sha(CONFIG_PATH)==CONFIG_EXPECTED_SHA,'E6 training configuration LOCK changed')
    c=read_json(CONFIG_PATH)
    need(c.get('status')=='LOCKED_PROTOCOL_E6_FROM_E5_2026-10-22', 'E6 configuration not locked')
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


def preflight(fold):
    need(1<=fold<=5,'Fold must be 1..5')
    cfg=load_config()
    lock_report=FOLDS/'02_e6_5fold_lock_report.json'
    check_report=read_json(AUDIT)
    fold_report=read_json(lock_report)
    need(fold_report.get('status')=='PASS_E6_FOLDS_LOCKED','E6 folds not locked')
    need(check_report.get('status')=='PASS_E6_DATALOADER_VALIDATION' and
         check_report.get('n_combinations_checked')==20 and
         check_report.get('model_weights_loaded') is False and
         check_report.get('test_pixel_read') is False and
         check_report.get('test_inference') is False,'E6 DataLoader audit not compatible')
    need(sha(lock_report)==check_report.get('source_fold_report_sha256')==
         cfg['locked_e6_fold_report_sha256'],'Stage2/3 E6 report hash broken')
    need(sha(FOLDS/'e6_development_1887_5fold_LOCKED.csv')==
         cfg['locked_e6_fold_manifest_sha256'],'E6 development fold manifest modified')
    need(sha(TEST_SOURCE)==fold_report['test_manifest_sha256']==
         check_report['e5_fixed_holdout_test_sha256']==cfg['e5_test_manifest_sha256'],
         'E5 fixed test manifest changed')
    need(sha(STAGE5/'processed_manifest.csv')==
         cfg['expected_hashes']['e5_processed_manifest_sha256'],
         'E5 processed PNG manifest changed')
    need(fold_report.get('no_patient_leakage') and
         fold_report.get('oof_each_development_image_exactly_once') and
         fold_report.get('oof_each_development_patient_exactly_once'),
         'E6 patient/OoF lock check failed')
    report_hashes=fold_report['output_sha256']
    for rel,h in report_hashes.items():
        relp=Path(rel)
        need(not relp.is_absolute() and '..' not in relp.parts,'Unsafe manifest relpath')
        p=FOLDS/relp
        need(p.is_file() and sha(p)==h,f'E6 fold artifact altered: {rel}')
    dev=read_csv(FOLDS/'e6_development_1887_5fold_LOCKED.csv')
    need(len(dev)==1887 and dev.patient_id.nunique()==1471 and
         dev.groupby('patient_id').cv_fold.nunique().eq(1).all(),
         'Development integrity/patient fold leak')
    need(not dev[KEY].duplicated().any() and dev.y_true.isin(('0','1')).all() and
         dev.e5_original_split.isin(('train','validation')).all(),
         'Development has duplicates, invalid labels, or E5 test rows')
    tr=read_csv(FOLDS/f'fold_{fold}/train.csv')
    va=read_csv(FOLDS/f'fold_{fold}/validation.csv')
    need(tr.split.eq('train').all() and va.split.eq('validation').all(),
         'Train/val manifest contains unexpected subset')
    need(len(tr)+len(va)==1887 and tr.patient_id.nunique()+va.patient_id.nunique()==1471,
         'Fold totals inconsistent')
    need(not set(tr.patient_id).intersection(va.patient_id), 'Fold patient leakage')
    need(tr.y_true.nunique()==2 and va.y_true.nunique()==2,'Missing label within fold')
    need(tr.cv_fold.ne(str(fold)).all() and va.cv_fold.eq(str(fold)).all(),
         'CV assigned validation fold wrong')
    dev_keys=set(map(tuple,dev[KEY].to_numpy()))
    tr_keys=set(map(tuple,tr[KEY].to_numpy()))
    va_keys=set(map(tuple,va[KEY].to_numpy()))
    need(not tr_keys.intersection(va_keys) and tr_keys|va_keys==dev_keys,
         'Fold image completeness/leak violation')
    for name,df in (('train',tr),('validation',va)):
        need(not df[KEY].duplicated().any(),'Duplicate fold image')
        for v in ('raw','clahe'):
            need(df[v+'_path'].nunique()==len(df),f'Duplicate PNG {name}/{v}')
            for rel in df[v+'_path']:
                need(safe_png_path(v,rel).is_file(),f'Missing development PNG {rel}')
    fps={'training_config':sha(CONFIG_PATH),
         'stage2_e6_folds':sha(lock_report),'stage3_e6_dataloader':sha(AUDIT),
         'e6_dev_locked':sha(FOLDS/'e6_development_1887_5fold_LOCKED.csv'),
         'e5_test_manifest':sha(TEST_SOURCE),
         'e5_processed_manifest':sha(STAGE5/'processed_manifest.csv'),
         'fold_train_manifest':sha(FOLDS/f'fold_{fold}/train.csv'),
         'fold_validation_manifest':sha(FOLDS/f'fold_{fold}/validation.csv'),
         'fold':fold,'train_images':len(tr),'validation_images':len(va)}
    print(f'LOCK PASS | E6 fold {fold} train={len(tr)} val={len(va)} '
          f'| E5 test pixels UNTOUCHED',flush=True)
    return cfg, {'train':tr.sort_values(KEY).reset_index(drop=True),
                 'validation':va.sort_values(KEY).reset_index(drop=True)}, fps


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
         'RAD-DINO model revision differs from E5')
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


def check_smoke(method, variant, fingerprints, gradient_checkpointing=False):
    # The same RAD-DINO architecture + PNG and microbatch as E5; previous full smoke exists.
    # E6 also has a separately runnable fold-specific smoke before complete training.
    p=E5_ROOT / 'audits' / '07_gpu_smoke' / f'{method}_{variant}.json'
    d=read_json(p)
    need(d.get('status')=='PASS_GPU_SMOKE' and d.get('method')==method and
         d.get('variant')==variant and
         d.get('gradient_checkpointing')==gradient_checkpointing,
         f'Prior GPU smoke test incompatible/missing: {p}')
    return d


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
        'status':'PASS_E6_FOLD_TRAINING_COMPLETE','experiment':'experiment6',
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
