from pathlib import Path
import json
import argparse
import hashlib
import sys

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(__file__).resolve().parent
CFG=SRC/'07_training_config_e7_LOCKED.json'
CFG_SHA='cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'


def inspect():
    if not CFG.exists() or hashlib.sha256(CFG.read_bytes()).hexdigest()!=CFG_SHA:
        raise RuntimeError('E7 four-model locked config is missing or altered')
    cfg=json.loads(CFG.read_text(encoding='utf-8'))
    if cfg['status']!='LOCKED_E7_FOUR_MODELS':
        raise RuntimeError('Incorrect E7 model protocol')
    done=0; invalid=0
    print('E7 5-FOLD TRAINING STATUS | TEST DATA NOT USED')
    print(f'{"MODEL":<18} {"FOLD":<5} {"STATUS":<14} {"BEST EPOCH":<11} {"VAL AUROC"}')
    for method in ('frozen','lora'):
        for variant in ('raw','clahe'):
            for fold in range(1,6):
                name=f'{method}_{variant}'
                folder=ROOT/'output'/'training'/f'{name}_fold_{fold}'
                summary=folder/'run_summary.json'
                latest=folder/'latest_training_state.pt'
                status='NOT STARTED';epoch='—';score='—'
                if summary.exists():
                    try:
                        d=json.loads(summary.read_text(encoding='utf-8'))
                        if not (d.get('status')=='PASS_E7_FOLD_TRAINING_COMPLETE' and
                                d.get('experiment')=='experiment7' and d.get('model')==name and
                                d.get('fold')==fold and d.get('locked_config_sha256')==CFG_SHA and
                                d.get('test_read') is False and
                                d.get('best_epoch',0)>0 and 0<=d.get('best_validation_auroc',-1)<=1 and
                                (folder/'training_history.csv').is_file()):
                            raise ValueError('Run summary and configuration do not match')
                        if method=='lora':
                            snapshot=d.get('best_snapshot_relative_path','')
                            if not snapshot or '..' in Path(snapshot).parts or Path(snapshot).is_absolute():
                                raise ValueError('Unsafe or missing LoRA best snapshot')
                            base=folder/snapshot
                            required=['adapter_model.safetensors','adapter_config.json',
                                      'best_classifier.pt','best_validation_predictions.csv']
                            if not all((base/f).is_file() for f in required):
                                raise ValueError('Incomplete LoRA checkpoint')
                        else:
                            if not all((folder/f).is_file() for f in (
                                    'best_classifier.pt','best_validation_predictions.csv')):
                                raise ValueError('Incomplete Frozen checkpoint')
                        status='COMPLETE';done+=1
                        epoch=str(d['best_epoch']); score=f'{d["best_validation_auroc"]:.4f}'
                    except (ValueError,TypeError,KeyError) as err:
                        status='INVALID';invalid+=1
                        print(f'INVALID {name} fold {fold}: {err}',file=sys.stderr)
                elif latest.is_file():
                    status='INCOMPLETE'
                elif folder.exists():
                    status='FOLDER ONLY'
                print(f'{name:<18} {fold:<5} {status:<14} {epoch:<11} {score}')
    print(f'COMPLETED: {done}/20 training runs')
    return done,invalid

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-complete',action='store_true')
    args=p.parse_args()
    try:
        done,invalid=inspect()
        if invalid or (args.require_complete and done!=20):sys.exit(1)
    except Exception as error:
        print(f'FAIL_E7_STATUS | {type(error).__name__}: {error}',file=sys.stderr)
        sys.exit(2)
