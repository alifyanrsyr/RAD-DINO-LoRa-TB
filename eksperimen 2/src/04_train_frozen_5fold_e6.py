from __future__ import annotations
import argparse, gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import e6_training_common as C


def get_features(variant,processor,device,offline,batch_size):
    """One immutable frozen backbone feature cache over 1887 DEVELOPMENT only.
    Permissible for CV: frozen pretrained backbone does no fitting and never reads test.
    """
    out=C.ROOT/'output'/'frozen_development_features'
    dest=out/f'{variant}_1887.pt'
    report=C.read_json(C.FOLDS/'02_e6_5fold_lock_report.json')
    provenance={'e6_dev_manifest_sha256':C.sha(C.FOLDS/'e6_development_1887_5fold_LOCKED.csv'),
                'e5_processed_sha256':C.sha(C.STAGE5/'processed_manifest.csv'),
                'e5_test_manifest_sha256':C.sha(C.TEST_SOURCE),
                'fold_report_sha256':C.sha(C.FOLDS/'02_e6_5fold_lock_report.json'),
                'variant':variant,'model':'microsoft/rad-dino'}
    rows=C.read_csv(C.FOLDS/'e6_development_1887_5fold_LOCKED.csv').sort_values(C.KEY).reset_index(drop=True)
    C.need(len(rows)==1887 and rows.patient_id.nunique()==1471 and
           rows.e5_original_split.isin(['train','validation']).all(),'Bad development features input')
    expected=(rows.source_archive+'::'+rows.archive_member_path).tolist()
    if dest.exists():
        obj=C.torch_load_trusted(dest)
        C.need(obj['provenance']==provenance and obj['keys']==expected and
               obj['labels'].tolist()==rows.y_true.astype(int).tolist() and
               tuple(obj['features'].shape)==(1887,768) and bool(torch.isfinite(obj['features']).all()),
               'Existing frozen CLS cache differs from locked development. Refuse reuse.')
        print('FROZEN DEVELOPMENT CLS CACHE PASS',dest,flush=True)
        return obj
    out.mkdir(parents=True,exist_ok=True)
    backbone=C.backbone_new(offline).to(device).eval()
    for p in backbone.parameters():p.requires_grad_(False)
    commit=getattr(backbone.config,'_commit_hash',None)
    feats=[];ys=[];keys=[]
    # Use new split column only for DataLoader's allowed-split audit. No use of E5 test.
    rows=rows.copy();rows['split']='train'
    loader=C.dataset_loader(rows,variant,processor,batch_size,shuffle=False)
    with torch.inference_mode():
        for j,(x,y,k,_) in enumerate(loader,1):
            z=backbone(pixel_values=x.to(device,non_blocking=True)).last_hidden_state[:,0,:].float().cpu()
            C.need(tuple(z.shape)==(len(y),768) and bool(torch.isfinite(z).all()),
                   'Invalid frozen CLS features')
            feats.append(z);ys.append(y);keys.extend(k)
            if j%100==0 or j==len(loader):
                print(f'FROZEN CLS {variant}: {j}/{len(loader)} batches',flush=True)
    obj={'provenance':provenance,'keys':keys,'labels':torch.cat(ys),
         'features':torch.cat(feats),'model_commit':commit}
    C.need(obj['keys']==expected and len(obj['labels'])==1887 and
           obj['labels'].tolist()==rows.y_true.astype(int).tolist(),
           'Development CLS cache ordering/labels mismatch')
    C.atomic_torch(dest,obj)
    del backbone,loader;gc.collect();torch.cuda.empty_cache()
    print('FROZEN CLS SAVED:',dest,flush=True)
    return obj


def select_features(obj,rows):
    lut={k:i for i,k in enumerate(obj['keys'])}
    keys=(rows.source_archive+'::'+rows.archive_member_path).tolist()
    C.need(len(set(keys))==len(keys) and all(k in lut for k in keys),
           'Fold images not in locked development feature cache')
    indices=[lut[k] for k in keys]
    x=obj['features'][indices];y=obj['labels'][indices]
    C.need(y.tolist()==rows.y_true.astype(int).tolist(),
           'Fold feature cache labels mismatch')
    return x,y


def eval_head(head,x,y,device):
    head.eval();loss_sum=0.;num_right=0;probs=[]
    with torch.inference_mode():
        for s in range(0,len(y),256):
            logits=head(x[s:s+256].to(device)).float()
            target=y[s:s+256].to(device)
            loss_sum+=float(nn.functional.cross_entropy(logits,target,reduction='sum').item())
            num_right+=int((logits.argmax(1)==target).sum().item())
            probs.extend(torch.softmax(logits,dim=1)[:,1].cpu().tolist())
    return loss_sum/len(y),num_right/len(y),np.asarray(probs,dtype=float)


def main(args):
    cfg,rows,fp=C.preflight(args.fold)
    C.need(args.feature_batch_size>=1,'Extractor batch size must be >= 1')
    C.need(not(args.smoke and args.resume),'Cannot combine --smoke and --resume')
    if args.check:
        print(f'PASS_E6_TRAIN_PREFLIGHT | frozen {args.variant} fold {args.fold} | no model loaded',flush=True)
        return
    C.seed_all(cfg['seed'])
    C.check_smoke('frozen',args.variant,fp,False)
    device=C.cuda_device()
    if args.smoke:
        processor=C.processor_new(args.offline_model)
        # FROZEN memory smoke: feature extraction + head gradient update, no output.
        backbone=C.backbone_new(args.offline_model).to(device).eval()
        for p in backbone.parameters(): p.requires_grad_(False)
        dl=C.dataset_loader(rows['train'].iloc[:1],args.variant,processor,1,False)
        x,y,_,_=next(iter(dl))
        with torch.inference_mode():
            cls=backbone(pixel_values=x.to(device)).last_hidden_state[:,0,:].float()
        # A clone OUTSIDE inference_mode is required for head-weight backpropagation.
        cls=cls.clone()
        head=nn.Linear(768,2).to(device)
        opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate_head'],weight_decay=cfg['weight_decay'])
        loss=nn.functional.cross_entropy(head(cls),y.to(device));loss.backward();opt.step()
        C.need(bool(torch.isfinite(loss)),'Non-finite Frozen smoke loss')
        print(f'PASS_E6_GPU_SMOKE | frozen/{args.variant}/fold{args.fold} '
              f'| loss={loss.item():.5f} | peak_allocated_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)
        return
    out=C.ROOT/'output/training'/f'frozen_{args.variant}_fold_{args.fold}'
    if args.resume:
        C.need(out.is_dir() and (out/'latest_training_state.pt').is_file() and
               not (out/'run_summary.json').exists(),'--resume requires unfinished fold checkpoint')
    else:
        C.need(not out.exists(),f'Folder output exists, will NOT overwrite: {out}')
        out.mkdir(parents=True)
    processor=C.processor_new(args.offline_model)
    feat=get_features(args.variant,processor,device,args.offline_model,args.feature_batch_size)
    x,y=select_features(feat,rows['train']);vx,vy=select_features(feat,rows['validation'])
    C.need(len(x)==len(rows['train']) and len(vx)==len(rows['validation']),
           'Feature fold dimensions mismatch')
    run_cfg={'experiment':'experiment6','method':'frozen','variant':args.variant,'fold':args.fold,
             'training_fingerprints':fp,'model_commit':feat['model_commit'],
             'head_batch_size':64,'extractor_physical_batch':args.feature_batch_size,
             'seed':42,'max_epochs':30,'patience':5,'min_delta':.001,
             'lr_head':.001,'weight_decay':.0001,'test_data_used':False,
             'best_checkpoint_metric':'fold_validation_AUROC'}
    if args.resume:
        C.need(C.read_json(out/'run_config.json')==run_cfg,'Run config changed; refuse resume')
    else:C.atomic_json(out/'run_config.json',run_cfg)
    head=nn.Linear(768,2).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate_head'],weight_decay=cfg['weight_decay'])
    last_epoch=0;best=float('-inf');best_epoch=0;stale=0;history=[]
    if args.resume:
        d=C.torch_load_trusted(out/'latest_training_state.pt')
        C.need(d['fingerprints']==fp and d['method']=='frozen' and d['fold']==args.fold and
               d['variant']==args.variant and d['model_commit']==feat['model_commit'],
               'Resume provenance/adapter mismatch')
        head.load_state_dict(d['head_state']);opt.load_state_dict(d['optimizer_state'])
        C.restore_rng(d['rng_state'])
        last_epoch=d['epoch'];best=d['best'];best_epoch=d['best_epoch'];stale=d['stale'];history=d['history']
        print(f'FROZEN RESUME from epoch {last_epoch}',flush=True)
    print(f'START E6 FROZEN fold={args.fold} {args.variant} train={len(y)} val={len(vy)} head_batch=64',flush=True)
    for ep in range(last_epoch+1,cfg['max_epochs']+1):
        head.train()
        generator=torch.Generator().manual_seed(cfg['seed']+ep)
        dl=DataLoader(TensorDataset(x,y),batch_size=64,shuffle=True,
                      generator=generator,drop_last=False,num_workers=0)
        for xb,yb in dl:
            xb=xb.to(device);yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(xb),yb)
            C.need(bool(torch.isfinite(loss)),'Non-finite training loss')
            loss.backward();opt.step()
        tl,ta,_=eval_head(head,x,y,device)
        vl,va,prob=eval_head(head,vx,vy,device)
        met=C.classification_metrics(vy.numpy(),prob)
        row={'epoch':ep,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':va,
             'val_auroc':met['auroc'],'val_macro_f1':met['macro_f1'],
             'val_balanced_accuracy':met['balanced_accuracy'],
             'val_sensitivity_dr':met['sensitivity_dr'],
             'val_specificity_ds':met['specificity_ds']}
        history.append(row)
        if met['auroc']>best+cfg['early_stopping_min_delta']:
            best=met['auroc'];best_epoch=ep;stale=0;flag='BEST'
            C.atomic_torch(out/'best_classifier.pt',
                {'classifier_state_dict':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                 'best_epoch':ep,'best_val_auroc':best,'fingerprints':fp,
                 'model_commit':feat['model_commit'],'fold':args.fold})
            C.atomic_csv(out/'best_validation_predictions.csv',
                         C.validation_prediction_csv(rows['validation'],prob))
            C.atomic_json(out/'best_validation_metrics.json',{'best_epoch':ep,**met,'val_loss':vl})
        else:stale+=1;flag=f'no_improve={stale}'
        C.atomic_csv(out/'training_history.csv',pd.DataFrame(history))
        C.atomic_torch(out/'latest_training_state.pt',
            {'method':'frozen','variant':args.variant,'fold':args.fold,'fingerprints':fp,
             'model_commit':feat['model_commit'],'epoch':ep,'best':best,
             'best_epoch':best_epoch,'stale':stale,'history':history,
             'head_state':head.state_dict(),'optimizer_state':opt.state_dict(),
             'rng_state':C.rng_state()})
        print(f'E6 FROZEN {args.variant} F{args.fold} EPOCH {ep:02d} | '
              f'train_loss={tl:.4f} train_acc={ta:.4f} | val_loss={vl:.4f} '
              f'val_acc={va:.4f} val_AUROC={met["auroc"]:.4f} | {flag}',flush=True)
        if stale>=cfg['early_stopping_patience']:
            print('Early stopping triggered',flush=True);break
    C.save_result_manifest(out,'frozen',args.variant,cfg,fp,best_epoch,best,history[-1]['epoch'],history,feat['model_commit'])
    print('PASS_E6_FOLD_TRAINING_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--variant',choices=['raw','clahe'],required=True)
    p.add_argument('--feature-batch-size',type=int,default=1,
                   help='Physical batch for no-grad CLS extraction only; head batch always 64')
    p.add_argument('--offline-model',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:main(a)
    except Exception as e:
        print(f'E6 FROZEN FAILED: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
