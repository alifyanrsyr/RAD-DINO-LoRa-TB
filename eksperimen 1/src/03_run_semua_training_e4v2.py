import argparse,sys,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=Path(__file__).resolve().parent
jobs=[(method,variant,fold) for method,variant in
      [('frozen','raw'),('frozen','clahe'),('lora','raw'),('lora','clahe')]
      for fold in range(1,6)]

def command(method,variant,fold,mode):
    f=SRC/f'03_train_{method}_5fold_e4v2.py'
    v=[sys.executable,str(f),'--fold',str(fold),'--variant',variant]
    return v+([mode] if mode else [])

def run(mode):
    for method,variant,fold in jobs:
        dest=ROOT/'output'/'training'/f'{method}_{variant}_fold_{fold}'
        summ=dest/'run_summary.json'; checkpoint=dest/'latest_training_state.pt'
        if summ.exists():
            j=json.loads(summ.read_text(encoding='utf-8'))
            if not (j.get('status')=='PASS_E4V2_FOLD_TRAINING_COMPLETE' and
                    j.get('method')==method and j.get('variant')==variant and
                    j.get('fold')==fold and j.get('experiment')=='experiment4v2'):
                raise RuntimeError(f'Invalid completion summary: {summ}')
            print('SKIP COMPLETE:',method,variant,fold,flush=True)
            continue
        if mode=='check':
            opt='--check'
        elif checkpoint.is_file():
            opt='--resume'
        elif dest.exists():
            raise RuntimeError(f'Unsafe partial directory, inspect before retry: {dest}')
        else:opt=None
        c=command(method,variant,fold,opt)
        print('RUN:', ' '.join(c),flush=True)
        subprocess.run(c,check=True,cwd=str(SRC))
    print('ALL JOBS CHECKED' if mode=='check' else 'ALL 20 RUNS COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    try:run('check' if p.parse_args().check else 'train')
    except Exception as e:
        print('FAILED:',e,file=sys.stderr);sys.exit(1)
