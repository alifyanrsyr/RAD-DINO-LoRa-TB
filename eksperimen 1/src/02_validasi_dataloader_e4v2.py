import argparse, sys
from pathlib import Path
import torch
import numpy as np
import e4v2_training_common as C

def main(args):
    # Preflight from Stage 1 only. No dependency on the Stage 2 report being created.
    prepared={}
    for fold in range(1,6):
        cfg,rows,fp=C.preflight(fold,require_dataloader_audit=False)
        prepared[fold]=(rows,fp)
    if args.check:
        print('DRY CHECK PASS_E4V2_DATALOADER_SOURCE | 1411 development images; no PNG pixels read')
        return
    processor=C.processor_new(args.offline_model)
    count=0
    records=[]
    for fold in range(1,6):
        rows,fp=prepared[fold]
        for subset in ('train','validation'):
            for variant in ('raw','clahe'):
                dl=C.dataset_loader(rows[subset],variant,processor,args.batch_size,False)
                n=0
                for x,y,keys,patients in dl:
                    C.need(tuple(x.shape[1:])==(3,518,518), 'Tensor dimensions mismatch')
                    C.need(torch.isfinite(x).all().item(), 'Nonfinite tensor')
                    C.need(y.min().item()>=0 and y.max().item()<=1,'Invalid class label')
                    n+=1
                    if n>=args.batches:break
                C.need(n==args.batches,'Insufficient loader batches')
                count+=1
                records.append({'fold':fold,'subset':subset,'variant':variant,
                                'batches_checked':n,'images_in_manifest':len(rows[subset])})
                print(f'PASS DataLoader fold={fold} {subset}/{variant}: {n} batches')
    lock=C.FOLDS/'01_e4_cv85_fold_lock_report.json'
    C.atomic_json(C.AUDIT,{
       'status':'PASS_E4V2_DATALOADER_VALIDATION',
       'checked_combinations':count,
       'source_fold_report_sha256':C.sha(lock),
       'original_e4_processed_manifest_sha256':C.sha(C.STAGE5/'processed_manifest.csv'),
       'test_pixel_accessed':False, 'test_inference_performed':False,
       'model_training_performed':False,
       'batch_size':args.batch_size,'batches_per_combination':args.batches,
       'records':records})
    print(f'PASS_E4V2_DATALOADER_VALIDATION | {count}/20 combinations | report={C.AUDIT}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    t=p.add_mutually_exclusive_group(required=True)
    t.add_argument('--check',action='store_true');t.add_argument('--run',action='store_true')
    p.add_argument('--offline-model',action='store_true')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--batches',type=int,default=2)
    args=p.parse_args()
    try:
        C.need(args.batch_size>=1 and args.batches>=1,'Invalid batch settings')
        main(args)
    except Exception as e:
        print(f'FAIL_E4V2_DATALOADER: {type(e).__name__}: {e}',file=sys.stderr)
        raise
