from __future__ import annotations
import argparse, shutil, sys, gc
import numpy as np
import pandas as pd
import torch
from torch import nn
import e6_training_common as C


def load_adapter_weights_trainable(backbone,state):
    available={k:p for k,p in backbone.named_parameters() if p.requires_grad}
    C.need(set(available)==set(state),'Resume LoRA adapter names changed')
    with torch.no_grad():
        for k,p in available.items():p.copy_(state[k].to(device=p.device,dtype=p.dtype))


def export_best_adapter(backbone, folder):
    target=folder/'best_adapter'
    stage=folder/'best_adapter_staging'
    backup=folder/'best_adapter_backup'
    C.need(not stage.exists() and not backup.exists(),
           'Adapter export interrupted; inspect staging/backup directory')
    backbone.save_pretrained(stage,safe_serialization=True)
    if target.exists():target.rename(backup)
    try:stage.rename(target)
    except Exception:
        if backup.exists() and not target.exists():backup.rename(target)
        raise
    if backup.exists():shutil.rmtree(backup)


def validate(backbone,head,rows,variant,processor,device):
    backbone.eval();head.eval()
    dl=C.dataset_loader(rows,variant,processor,1,False)
    loss_sum=0.;right=0;labels=[];keys=[];probs=[]
    with torch.inference_mode():
        for x,y,k,_ in dl:
            with torch.cuda.amp.autocast(enabled=True):
                cls=backbone(pixel_values=x.to(device,non_blocking=True)).last_hidden_state[:,0,:]
                logits=head(cls)
            logits=logits.float();yb=y.to(device)
            loss=nn.functional.cross_entropy(logits,yb,reduction='sum')
            C.need(bool(torch.isfinite(loss).item()),'Nonfinite LoRA validation loss')
            loss_sum+=float(loss.item())
            right+=int((logits.argmax(1)==yb).sum().item())
            probs.extend(torch.softmax(logits,dim=1)[:,1].cpu().tolist())
            labels.extend(y.tolist());keys.extend(k)
    expected=(rows.source_archive+'::'+rows.archive_member_path).tolist()
    C.need(keys==expected and labels==rows.y_true.astype(int).tolist(),
           'Fold validation labels/order mismatch')
    met=C.classification_metrics(labels,probs)
    return loss_sum/len(rows),right/len(rows),np.asarray(probs,dtype=float),met


def smoke_test(backbone,head,processor,device,rows,variant,cfg):
    # One full optimizer step over first 16 *real training images*; no artifacts written.
    backbone.train();head.train()
    trainable=[p for p in backbone.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':trainable,'lr':cfg['learning_rate_lora']},
                           {'params':list(head.parameters()),'lr':cfg['learning_rate_head']}],
                          weight_decay=cfg['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=1024.0)
    dl=C.dataset_loader(rows.iloc[:16],variant,processor,1,False)
    opt.zero_grad(set_to_none=True)
    for i,(x,y,_,_) in enumerate(dl,1):
        with torch.cuda.amp.autocast(enabled=True):
            z=backbone(pixel_values=x.to(device)).last_hidden_state[:,0,:]
            logits=head(z)
            loss=nn.functional.cross_entropy(logits.float(),y.to(device))
        C.need(bool(torch.isfinite(loss).item()),f'Nonfinite smoke loss micro-batch {i}')
        scaler.scale(loss/16).backward()
    C.need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable),
           'LoRA adapter gradients missing or nonfinite')
    before=scaler.get_scale();scaler.step(opt);scaler.update()
    C.need(scaler.get_scale()>=before,'AMP skipped LoRA smoke optimizer step')
    print(f'PASS_E6_GPU_SMOKE | LoRA {variant} | 16 micro-batches '
          f'| peak_allocated_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}',flush=True)


def main(a):
    cfg,rows,fp=C.preflight(a.fold)
    if a.check:
        print(f'PASS_E6_TRAIN_PREFLIGHT | LoRA {a.variant} fold {a.fold} | no model loaded',flush=True)
        return
    C.need(not (a.smoke and a.resume),'Cannot combine smoke and resume')
    C.seed_all(cfg['seed'])
    C.check_smoke('lora',a.variant,fp,a.gradient_checkpointing)
    device=C.cuda_device()
    out=C.ROOT/'output/training'/f'lora_{a.variant}_fold_{a.fold}'
    if not a.smoke:
        if a.resume:
            C.need(out.is_dir() and (out/'latest_training_state.pt').is_file() and
                   not (out/'run_summary.json').exists(),
                   'Resume requires incomplete fold checkpoint')
        else:
            C.need(not out.exists(),f'Output exists; refuse to overwrite: {out}')
            out.mkdir(parents=True)
    processor=C.processor_new(a.offline_model)
    backbone=C.backbone_new(a.offline_model)
    model_commit=getattr(backbone.config,'_commit_hash',None)
    if a.gradient_checkpointing:
        C.need(hasattr(backbone,'gradient_checkpointing_enable'),
               'Backbone does not support gradient checkpointing')
        backbone.gradient_checkpointing_enable()
    backbone=C.lora_wrap(backbone).to(device)
    head=nn.Linear(768,2).to(device)
    C.need(sum(p.numel() for p in head.parameters() if p.requires_grad)==1538,
           'Expected classifier head has 1538 parameters')
    if a.smoke:
        smoke_test(backbone,head,processor,device,rows['train'],a.variant,cfg)
        return
    run_cfg={
        'experiment':'experiment6','method':'lora','variant':a.variant,'fold':a.fold,
        'training_fingerprints':fp,'model_commit':model_commit,
        'physical_batch':1,'gradient_accumulation':16,'effective_batch':16,
        'gradient_checkpointing':a.gradient_checkpointing,
        'optimizer':'AdamW','lr_lora':.0001,'lr_head':.001,'weight_decay':.0001,
        'lora_layers_0_based':[8,9,10,11],'lora_projections':['query','value'],
        'lora_r':8,'lora_alpha':16,'lora_dropout':.05,'max_epochs':30,
        'patience':5,'min_delta':.001,'seed':42,'test_data_used':False,
        'best_checkpoint_selection':'fold_validation_AUROC'
    }
    if a.resume:
        C.need(C.read_json(out/'run_config.json')==run_cfg,'Run config changed; refuse resume')
    else:C.atomic_json(out/'run_config.json',run_cfg)
    trainable=[p for p in backbone.parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':trainable,'lr':cfg['learning_rate_lora']},
                           {'params':list(head.parameters()),'lr':cfg['learning_rate_head']}],
                          weight_decay=cfg['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=1024.0)
    last_epoch=0;best=float('-inf');best_epoch=0;stale=0;history=[]
    if a.resume:
        saved=C.torch_load_trusted(out/'latest_training_state.pt')
        C.need(saved['method']=='lora' and saved['fold']==a.fold and
               saved['variant']==a.variant and saved['fingerprints']==fp and
               saved['model_commit']==model_commit and
               saved['gradient_checkpointing']==a.gradient_checkpointing,
               'LoRA resume provenance/memory mode mismatch')
        load_adapter_weights_trainable(backbone,saved['trainable_lora_state'])
        head.load_state_dict(saved['head_state'])
        opt.load_state_dict(saved['optimizer_state'])
        scaler.load_state_dict(saved['scaler_state'])
        C.restore_rng(saved['rng_state'])
        last_epoch=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch']
        stale=saved['stale'];history=saved['history']
        print(f'RESUMED LoRA fold {a.fold} after epoch {last_epoch}',flush=True)
    print(f'START E6 LORA fold={a.fold} variant={a.variant} physical_batch=1 '
          f'accumulation=16 train={len(rows["train"])} '
          f'validation={len(rows["validation"])} fp16=True',flush=True)
    for ep in range(last_epoch+1,cfg['max_epochs']+1):
        backbone.train();head.train();opt.zero_grad(set_to_none=True)
        dl=C.dataset_loader(rows['train'],a.variant,processor,1,True,ep)
        n=len(dl)
        loss_sum=0.;correct=0;skips=0;steps=0
        for i,(x,y,_,_) in enumerate(dl):
            start=(i//16)*16
            group_size=min(16,n-start)
            C.need(group_size>=1,'Invalid gradient accumulation group')
            xb=x.to(device,non_blocking=True);yb=y.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                cls=backbone(pixel_values=xb).last_hidden_state[:,0,:]
                logits=head(cls)
                loss=nn.functional.cross_entropy(logits.float(),yb)
            C.need(bool(torch.isfinite(loss).item()),
                   f'Nonfinite LoRA loss fold {a.fold} ep{ep}/image{i}')
            scaler.scale(loss/group_size).backward()
            loss_sum+=float(loss.item())
            correct+=int((logits.detach().float().argmax(1)==yb).sum().item())
            if (i+1)%16==0 or i+1==n:
                if ep==1 and i+1==16:
                    C.need(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                               for p in trainable),'LoRA gradients absent/invalid')
                before=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()<before:skips+=1
                else:steps+=1
                opt.zero_grad(set_to_none=True)
            if (i+1)%200==0 or i+1==n:
                print(f'  E6 LoRA {a.variant} F{a.fold} epoch {ep:02d}: '
                      f'micro-batch {i+1}/{n} | steps={steps} | amp_skips={skips}',flush=True)
        C.need(skips<=3,f'Too many AMP skipped steps: {skips}')
        val_loss,val_acc,probs,met=validate(backbone,head,rows['validation'],
                                             a.variant,processor,device)
        row={'epoch':ep,'train_loss':loss_sum/n,'train_accuracy':correct/n,
             'val_loss':val_loss,'val_accuracy':val_acc,'val_auroc':met['auroc'],
             'val_macro_f1':met['macro_f1'],
             'val_balanced_accuracy':met['balanced_accuracy'],
             'val_sensitivity_dr':met['sensitivity_dr'],
             'val_specificity_ds':met['specificity_ds'],
             'optimizer_steps':steps,'fp16_skipped_steps':skips}
        history.append(row)
        if met['auroc']>best+cfg['early_stopping_min_delta']:
            export_best_adapter(backbone,out)
            C.atomic_torch(out/'best_classifier.pt',{
                'classifier_state_dict':{k:v.detach().cpu().clone()
                                         for k,v in head.state_dict().items()},
                'best_epoch':ep,'best_val_auroc':met['auroc'],
                'model_commit':model_commit,'fold':a.fold,'fingerprints':fp})
            C.atomic_csv(out/'best_validation_predictions.csv',
                         C.validation_prediction_csv(rows['validation'],probs))
            C.atomic_json(out/'best_validation_metrics.json',
                          {'best_epoch':ep,**met,'val_loss':val_loss})
            best=met['auroc'];best_epoch=ep;stale=0;flag='BEST'
        else:stale+=1;flag=f'no_improve={stale}'
        C.atomic_csv(out/'training_history.csv',pd.DataFrame(history))
        params={k:p.detach().cpu().clone() for k,p in backbone.named_parameters()
                if p.requires_grad}
        C.atomic_torch(out/'latest_training_state.pt',{
            'method':'lora','variant':a.variant,'fold':a.fold,'fingerprints':fp,
            'model_commit':model_commit,'gradient_checkpointing':a.gradient_checkpointing,
            'epoch':ep,'best':best,'best_epoch':best_epoch,'stale':stale,
            'history':history,'trainable_lora_state':params,
            'head_state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            'optimizer_state':opt.state_dict(),'scaler_state':scaler.state_dict(),
            'rng_state':C.rng_state()})
        print(f'E6 LoRA {a.variant} F{a.fold} EPOCH {ep:02d} | '
              f'train_loss {loss_sum/n:.4f} train_acc {correct/n:.4f} | '
              f'val_loss {val_loss:.4f} val_acc {val_acc:.4f} '
              f'val_AUROC {met["auroc"]:.4f} | {flag}',flush=True)
        if stale>=cfg['early_stopping_patience']:
            print('Early stopping triggered',flush=True);break
    C.save_result_manifest(out,'lora',a.variant,cfg,fp,best_epoch,best,
                           history[-1]['epoch'],history,model_commit)
    print('PASS_E6_FOLD_TRAINING_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--variant',choices=['raw','clahe'],required=True)
    p.add_argument('--offline-model',action='store_true')
    p.add_argument('--gradient-checkpointing',action='store_true',
                   help='Only if matching E5 GPU smoke tested with same memory mode')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:main(a)
    except Exception as e:
        print(f'E6 LORA TRAIN FAILED: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise
