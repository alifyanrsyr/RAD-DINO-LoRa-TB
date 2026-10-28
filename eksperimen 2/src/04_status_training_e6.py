from pathlib import Path
import json, csv
ROOT=Path(__file__).resolve().parents[1]
models=[('frozen','raw'),('frozen','clahe'),('lora','raw'),('lora','clahe')]
finished=0
print('E6 CROSS-VALIDATION TRAINING STATUS | E5 test holdout untouched')
print(f'{"MODEL":<17} {"FOLD":<5} {"STATUS":<15} {"BEST EPOCH":<11} {"VAL AUROC":<10}')
for method,variant in models:
    for fold in range(1,6):
        d=ROOT/'output'/'training'/f'{method}_{variant}_fold_{fold}'
        summary=d/'run_summary.json'
        progress=d/'latest_training_state.pt'
        if summary.exists():
            r=json.loads(summary.read_text(encoding='utf-8'))
            if (r.get('status')=='PASS_E6_FOLD_TRAINING_COMPLETE' and
                r.get('method')==method and r.get('variant')==variant and r.get('fold')==fold and
                not r.get('test_pixel_read') and not r.get('test_inference')):
                status='COMPLETE'; finished+=1
                ep=r.get('best_epoch','—');au=f'{r.get("best_validation_auroc",0):.4f}'
            else:status='BAD SUMMARY';ep='—';au='—'
        elif progress.exists():status='INCOMPLETE';ep='—';au='—'
        elif d.exists():status='INCOMPLETE';ep='—';au='—'
        else:status='NOT STARTED';ep='—';au='—'
        print(f'{method+"_"+variant:<17} {fold:<5} {status:<15} {str(ep):<11} {au:<10}')
print(f'COMPLETED: {finished}/20 training runs')
