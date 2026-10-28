from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = [('M1','frozen_raw','Frozen RAW'),('M2','frozen_clahe','Frozen CLAHE'),
          ('M3','lora_raw','LoRA RAW'),('M4','lora_clahe','LoRA CLAHE')]
CODES=[m[0] for m in MODELS]
MODEL_NAME={code:label for code,_,label in MODELS}
NAME_TO_CODE={model:code for code,model,_ in MODELS}
METRICS=['auroc','macro_f1','balanced_accuracy','sensitivity_dr','specificity_ds']
LABELS=['AUROC','Macro-F1','Balanced accuracy','Sensitivity DR-TB','Specificity DS-TB']
PLOT_NAMES=[
 '01_e7_holdout_auroc.png',
 '02_e7_macro_f1_balanced_accuracy.png',
 '03_e7_confusion_matrices.png',
 '04_e7_sensitivity_specificity.png',
 '05_e7_oof_vs_holdout_auroc.png',
 '06_e7_fold_auroc_mean_sd.png',
 '07_e7_primary_model_metrics.png',
 '08_e7_primary_oof_holdout_differences.png',
 '09_e7_holdout_metrics_table.png',
 '10_roc_curve_e7.png',
] + [f'{11+i*2:02d}_{model}_loss.png' for i,model in enumerate(NAME_TO_CODE)] + [
 f'{12+i*2:02d}_{model}_accuracy.png' for i,model in enumerate(NAME_TO_CODE)]
TITLE_MAP={
 PLOT_NAMES[0]:'Holdout Test AUROC',
 PLOT_NAMES[1]:'Macro-F1 and Balanced Accuracy',
 PLOT_NAMES[2]:'Confusion Matrix',
 PLOT_NAMES[3]:'Sensitivity and Specificity',
 PLOT_NAMES[4]:'OOF and Test AUROC',
 PLOT_NAMES[5]:'AUROC Across Five Folds',
 PLOT_NAMES[6]:'Classification Performance',
 PLOT_NAMES[7]:'OOF and Holdout Performance Differences',
 PLOT_NAMES[8]:'Holdout Test Performance',
 PLOT_NAMES[9]:'Receiver Operating Characteristic (ROC) Curve',
}
for _n in PLOT_NAMES[10:]:
 TITLE_MAP[_n]='Training and Validation Loss' if _n.endswith('_loss.png') else 'Training and Validation Accuracy'
SOURCE_NAMES={
 'e7_test_csv':'12_e7_ensemble_test_comparison.csv',
 'e7_test_lock':'12_e7_ensemble_holdout_test_LOCKED.json',
 'e7_oof_csv':'08_e7_oof_comparison_4models.csv',
 'e7_oof_lock':'08_e7_oof_model_selection_LOCKED.json',
 'e7_fold_overfit':'10_e7_overfitting_per_fold.csv',
}

def need(ok, message):
    if not ok: raise RuntimeError(str(message))

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def load_json(p:Path):
    need(p.is_file(),f'JSON tidak ditemukan: {p}')
    return json.loads(p.read_text(encoding='utf-8-sig'))

def close(a,b,eps=1e-9):
    return math.isclose(float(a),float(b),abs_tol=eps,rel_tol=eps)

def _norm(name):
    return re.sub(r'[^a-z0-9]+','_',str(name).strip().lower()).strip('_')

def _model_code(raw):
    s=_norm(raw)
    if s in {'m1','frozen_raw','frozen_raw_m1'} or ('frozen' in s and 'raw' in s): return 'M1'
    if s in {'m2','frozen_clahe'} or ('frozen' in s and 'clahe' in s): return 'M2'
    if s in {'m3','lora_raw'} or ('lora' in s and 'raw' in s): return 'M3'
    if s in {'m4','lora_clahe'} or ('lora' in s and 'clahe' in s): return 'M4'
    return None

def locate(root:Path, name:str, optional=False)->Path|None:
    """Cari filename tepat. Salinan dengan isi bertentangan harus dihentikan."""
    need(root.is_dir(),f'Folder tidak ditemukan: {root}')
    known=list((root/'output').rglob(name)) if (root/'output').is_dir() else []
    if not known:
        known=list(root.rglob(name))
    found=sorted([p for p in known if p.is_file()],key=lambda p:(len(p.parts),str(p)))
    if not found:
        if optional: return None
        raise RuntimeError(f'Berkas wajib tidak ditemukan: {name} di {root}')
    if len(found)>1:
        all_hash={sha(p) for p in found}
        need(len(all_hash)==1,
             f'Salinan {name} berbeda SHA256; pilih satu berkas yang benar dan pindahkan salinan bertentangan:\n'+
             '\n'.join(map(str,found)))
    return found[0]

def col(df, choices, required=True):
    lookup={_norm(k):k for k in df.columns}
    for ch in choices:
        if _norm(ch) in lookup: return lookup[_norm(ch)]
    if required: raise RuntimeError('Kolom tidak ada: '+str(choices)+' | tersedia: '+str(list(df.columns)))
    return None

def read_metrics(p,split):
    """Normalisasi CSV E7 menjadi skema metrik akademis yang digunakan renderer."""
    d=pd.read_csv(p,encoding='utf-8-sig',low_memory=False)
    cname=col(d,['model_code','model','model_name','model_key','configuration'])
    d['model_code']=[_model_code(z) for z in d[cname]]
    need(len(d)==4 and set(d.model_code)==set(CODES),'Harus tepat empat model M1--M4: '+str(p))
    d=d.set_index('model_code',drop=False).loc[CODES].copy()
    keys={
      'auroc':(['oof_auroc','pooled_oof_auroc','pooled_auroc','auroc','auc'] if split=='oof'
               else ['auroc','test_auroc','ensemble_auroc','holdout_test_auroc']),
      'macro_f1':['oof_macro_f1','macro_f1','f1_macro'],
      'balanced_accuracy':['oof_balanced_accuracy','balanced_accuracy','balanced_acc'],
      'sensitivity_dr':['oof_sensitivity_dr','sensitivity_dr','sensitivity','recall_dr'],
      'specificity_ds':['oof_specificity_ds','specificity_ds','specificity'],
      'accuracy':['oof_accuracy','accuracy','acc'],
      'tn':['oof_tn','tn'],'fp':['oof_fp','fp'],'fn':['oof_fn','fn'],'tp':['oof_tp','tp'],
    }
    for key, choices in keys.items():
        k=col(d,choices,required=(key in METRICS or split=='test'))
        if k is not None:
            d[key]=pd.to_numeric(d[k],errors='raise')
            need(np.isfinite(d[key]).all(),f'Invalid numeric {key}: {p}')
    for metric in METRICS:
        need(d[metric].between(0,1).all(),f'Out-of-range {metric}: {p}')
    if split=='test':
        for code,row in d.iterrows():
            tn,fp,fn,tp=[int(row[k]) for k in ('tn','fp','fn','tp')]
            need((tn+fp,fn+tp)==(166,167),f'Confusion totals invalid {code}: {p}')
            need(close(row['sensitivity_dr'],tp/167,1e-6),f'Sensitivity mismatch {code}: {p}')
            need(close(row['specificity_ds'],tn/166,1e-6),f'Specificity mismatch {code}: {p}')
            need(close(row['balanced_accuracy'],(tp/167+tn/166)/2,1e-6),f'Balanced accuracy mismatch {code}: {p}')
            if 'accuracy' in d:
                need(close(row['accuracy'],(tn+tp)/333,1e-6),f'Accuracy mismatch {code}: {p}')
        th=col(d,['threshold','decision_threshold'],required=False)
        if th: need(all(close(v,0.5) for v in d[th]),'Threshold test primer bukan 0.50')
    if split=='oof':
        mean=col(d,['fold_auroc_mean'],required=False)
        sd=col(d,['fold_auroc_sd_sample','fold_auroc_std','fold_auroc_sd'],required=False)
        if mean: d['fold_auroc_mean']=pd.to_numeric(d[mean],errors='raise')
        if sd: d['fold_auroc_sd_sample']=pd.to_numeric(d[sd],errors='raise')
        d['oof_auroc']=d['auroc']
    return d

def _deep_values(obj,key):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if k==key: yield v
            yield from _deep_values(v,key)
    elif isinstance(obj,list):
        for v in obj: yield from _deep_values(v,key)

def verify_lock_csv(lock,path):
    """Verifikasi checksum LOCKED bila lock menyimpan pasangan filename/hash."""
    name=path.name
    hashes=set()
    for key in ('output_sha256','output_files_sha256','files_sha256'):
        for mapping in _deep_values(lock,key):
            if isinstance(mapping,dict):
                for k,v in mapping.items():
                    if Path(str(k)).name==name and isinstance(v,str) and re.fullmatch('[a-fA-F0-9]{64}',v):
                        hashes.add(v.lower())
    if hashes:
        need(len(hashes)==1 and sha(path).lower() in hashes,f'SHA-256 bertentangan untuk {path}')
        return 'hash_verified'
    return 'no_registered_hash'   # Dilaporkan, bukan mengarang status verifikasi

def _optional_primary(lock,label):
    matches=[]
    for key in ('primary_model','primary_model_code','selected_model'):
        if key in lock and isinstance(lock[key],str): matches.append((key,lock[key]))
    for key,v in matches:
        need(_model_code(v)=='M4',f'{label}: {key} bukan M4: {v}')
    if lock.get('test_based_model_selection') is True:
        raise RuntimeError(f'{label}: terdeteksi model selection berdasarkan test')
    for k in ('status',):
        if k in lock: need(str(lock[k]).upper().startswith('PASS'),f'{label}: status bukan PASS: {lock[k]}')

def discover_sources(root):
    paths={k:locate(root,n,optional=(k=='e7_fold_overfit')) for k,n in SOURCE_NAMES.items()}
    locks={k:load_json(v) for k,v in paths.items() if k.endswith('_lock')}
    for name,lock in locks.items(): _optional_primary(lock,name)
    hash_status={}
    for key in ('e7_test_csv','e7_oof_csv'):
        hash_status[key]=verify_lock_csv(locks[key[:-3]+'lock'],paths[key])
    if paths['e7_fold_overfit'] is not None:
        hash_status['e7_fold_overfit']=verify_lock_csv(locks['e7_oof_lock'],paths['e7_fold_overfit'])
    test=read_metrics(paths['e7_test_csv'],'test')
    oof=read_metrics(paths['e7_oof_csv'],'oof')
    need(close(float(oof.loc['M4','auroc']),0.733542,1e-6),
         'OOF AUROC model utama E7 tidak sesuai nilai LOCKED')
    need(close(float(test.loc['M4','auroc']),0.770327,1e-6),
         'Holdout AUROC model utama E7 tidak sesuai nilai LOCKED')
    # E7 primary decision remains M4. No selection by holdout results.
    return paths,locks,hash_status,test,oof

def gather_fold_statistics(oof,e7root,paths):
    if 'fold_auroc_mean' in oof and 'fold_auroc_sd_sample' in oof:
        if np.isfinite(oof[['fold_auroc_mean','fold_auroc_sd_sample']].astype(float)).all().all():
            return oof
    p=paths['e7_fold_overfit']
    need(p is not None,'Fold mean ± SD tidak ada di OOF; butuh 10_e7_overfitting_per_fold.csv')
    df=pd.read_csv(p)
    m=col(df,['model','model_code','model_key'])
    f=col(df,['fold','fold_id'])
    auc=col(df,['best_val_auroc','best_validation_auroc','best_auroc','best_validation_metrics_auroc'])
    df=df.assign(model_code=[_model_code(z) for z in df[m]],fold=pd.to_numeric(df[f]),fold_auroc=pd.to_numeric(df[auc]))
    need(len(df)==20 and set(df.model_code)==set(CODES),'Data AUROC per-fold harus 4×5')
    for code in CODES:
        d=df[df.model_code==code]
        need(set(d.fold)==set(range(1,6)),f'Fold tidak lengkap untuk {code}')
        vals=d.fold_auroc.to_numpy(dtype=float)
        need(np.isfinite(vals).all() and np.all((0<=vals)&(vals<=1)),f'AUROC fold invalid {code}')
        oof.loc[code,'fold_auroc_mean']=float(np.mean(vals))
        oof.loc[code,'fold_auroc_sd_sample']=float(np.std(vals,ddof=1))
    return oof

def _history_file(root:Path, model:str, fold:int)->Path:
    run=f'{model}_fold_{fold}'
    direct=root/'output'/'training'/run/'training_history.csv'
    if direct.is_file(): return direct
    base=root/'output'/'training'
    matches=[]
    if base.is_dir():
        matches=[p for p in base.rglob('training_history.csv') if run in '/'.join(p.parts).lower()]
    if not matches:
        matches=[p for p in (root/'output').rglob('training_history.csv')
                 if run in '/'.join(p.parts).lower()]
    need(matches,'Riwayat training tidak ditemukan: '+run+'/training_history.csv')
    matches=sorted(matches,key=lambda p:(len(p.parts),str(p)))
    need(len({sha(p) for p in matches})==1,
         f'Terdapat riwayat training berbeda untuk {run}:\n'+'\n'.join(map(str,matches)))
    return matches[0]

def _pred_file(root:Path,model:str)->Path:
    candidates=[f'12_e7_test_ensemble_predictions_{model}.csv',
                f'12_test_ensemble_predictions_{model}.csv',
                f'12_e7_ensemble_test_predictions_{model}.csv',
                f'12_e7_ensemble_predictions_{model}.csv']
    paths=[]
    for name in candidates:
        path=locate(root,name,optional=True)
        if path: paths.append(path)
    if not paths:
        folder=root/'output'/'testing'
        if folder.is_dir():
            paths=[p for p in folder.rglob('*.csv')
                   if re.search(r'(^|_)predictions?_',p.name.lower()) and
                   model in p.name.lower() and ('12_' in p.name or 'e7' in p.name.lower())]
    need(paths,f'File prediksi holdout E7 tidak ditemukan untuk {model}. ROC tidak boleh direkonstruksi dari AUROC agregat.')
    paths=sorted(paths,key=lambda p:(len(p.parts),str(p)))
    need(len({sha(p) for p in paths})==1,
         f'Ditemukan kandidat prediksi berbeda untuk {model}:\n'+'\n'.join(map(str,paths)))
    return paths[0]

def load_auxiliary(e7root,ooflock,testlock,e7):
    histories={}; preds={}; hist_hash={}; prediction_hash={}; integrity={}
    for code,model,_ in MODELS:
        histories[model]=[]
        for fold in range(1,6):
            p=_history_file(e7root,model,fold)
            run=f'{model}_fold_{fold}'
            frame=pd.read_csv(p)
            need({'epoch','train_loss','val_loss','train_accuracy','val_accuracy'}.issubset(frame.columns),
                 f'Kolom training tidak lengkap: {p}')
            need(len(frame)>=2 and frame.epoch.is_unique and float(frame.epoch.iloc[0])==1,f'Epoch invalid: {p}')
            for k in ('train_loss','val_loss','train_accuracy','val_accuracy'):
                need(np.isfinite(pd.to_numeric(frame[k],errors='coerce')).all(),f'Nonfinite {k}: {p}')
            histories[model].append(frame)
            hist_hash[run]=sha(p)
            expected=(ooflock.get('run_artifact_sha256',{}).get(run,{})
                          .get('training_history.csv'))
            if expected:
                need(hist_hash[run]==expected,f'Training history hash berbeda: {p}')
                integrity[run]='hash_verified'
            else: integrity[run]='no_registered_hash'
        p=_pred_file(e7root,model)
        frame=pd.read_csv(p)
        truth_name=col(frame,['y_true','true_label','label','target'])
        prob_name=col(frame,['p_dr_mean_5fold','p_dr_ensemble','prob_dr','probability_dr','y_prob_dr','probability'])
        need(len(frame)==333,f'Prediksi {model} bukan 333 citra: {p}')
        truth=pd.to_numeric(frame[truth_name],errors='raise').to_numpy(dtype=int)
        prob=pd.to_numeric(frame[prob_name],errors='raise').to_numpy(dtype=float)
        need(set(truth)=={0,1} and (truth==0).sum()==166 and (truth==1).sum()==167,
             f'Label prediksi test tidak cocok {model}')
        need(np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),f'Probabilitas invalid: {p}')
        from sklearn.metrics import roc_auc_score
        score=roc_auc_score(truth,prob)
        need(close(score,e7.loc[code,'auroc'],2e-6),
             f'ROC prediksi vs test ringkasan berbeda: {code}: {score} != {e7.loc[code,"auroc"]}')
        preds[code]=(truth,prob)
        prediction_hash[p.name]=sha(p)
        integrity[p.name]=verify_lock_csv(testlock,p)
    return histories,preds,{'history':hist_hash,'roc':prediction_hash,'integrity':integrity}


def render_roc(stage,preds,e7):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig,ax=plt.subplots(figsize=(12,8))
    colors=['#426AA3','#DD9446','#40856D','#95549E']
    for (code,_,label),color in zip(MODELS,colors):
        truth,prob=preds[code]
        score=roc_auc_score(truth,prob)
        need(abs(score-float(e7.loc[code,'auroc']))<1e-9,
             f'ROC AUROC mismatch for {code}')
        fpr,tpr,_=roc_curve(truth,prob)
        ax.plot(fpr,tpr,label=f'{code} — {label} (AUROC {score:.3f})',lw=2,color=color)
    ax.plot([0,1],[0,1],ls='--',lw=1,color='gray')
    ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    ax.set_xlabel('False Positive Rate');ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',fontsize=17,pad=18)
    ax.legend(loc='lower right',fontsize=10);ax.grid(alpha=.18)
    fig.savefig(stage/'10_roc_curve_e7.png',dpi=140,bbox_inches='tight',pad_inches=.3)
    plt.close(fig)

def render_training_histories(stage,histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    for idx,model in enumerate(['frozen_raw','frozen_clahe','lora_raw','lora_clahe']):
        folds=histories[model]
        for suffix,train_col,val_col,title,ylabel in [
            ('loss','train_loss','val_loss','Training and Validation Loss','Cross-Entropy Loss'),
            ('accuracy','train_accuracy','val_accuracy','Training and Validation Accuracy','Accuracy')]:
            fig,ax=plt.subplots(figsize=(13,8))
            for fold,df in enumerate(folds,1):
                color=cm.tab10((fold-1)/10)
                ax.plot(df.epoch,df[train_col],color=color,lw=1.7,
                        label=f'Fold {fold} — Training')
                ax.plot(df.epoch,df[val_col],color=color,lw=1.7,ls='--',
                        label=f'Fold {fold} — Validation')
            ax.set_xlabel('Epoch');ax.set_ylabel(ylabel)
            ax.set_title(title,fontsize=17,pad=18)
            ax.grid(alpha=.18)
            ax.legend(ncol=2,fontsize=9,loc='best')
            if suffix=='accuracy': ax.set_ylim(0,1.02)
            filename=f'{11+idx*2+(suffix=="accuracy"):02d}_{model}_{suffix}.png'
            fig.savefig(stage/filename,dpi=130,bbox_inches='tight',pad_inches=.3)
            plt.close(fig)

def render_plots(stage,test,oof):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    x=np.arange(4); names=[MODEL_NAME[z] for z in CODES]
    C1='#436FA6'; C2='#D38D48'; GREEN='#40856D'; RED='#AE5F5F'
    def save(name,fig):
        fig.savefig(stage/name,dpi=130,bbox_inches='tight',pad_inches=.35)
        plt.close(fig)
    def annotate(ax,bars,fmt='{:.3f}',offset=.008):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+offset,
                    fmt.format(bar.get_height()),ha='center',va='bottom',fontsize=10)
    def setup(ax,top=0.92):
        ax.set_ylim(0,top);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)

    # 01 - identical overall bar styling to the E6 comparison chart; E7 models only.
    fig,ax=plt.subplots(figsize=(16,9))
    bars=ax.bar(x,test.auroc.astype(float),.57,color=C1)
    annotate(ax,bars)
    ax.set_xticks(x,names);ax.set_ylabel('AUROC');setup(ax)
    ax.set_title('Holdout Test AUROC',pad=20,fontweight='bold',fontsize=17)
    save(PLOT_NAMES[0],fig)

    # 02 - paired-panel E6 layout, one E7 series in each panel.
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    for ax,metric,title in zip(axes,['macro_f1','balanced_accuracy'],
                                ['Macro-F1','Balanced Accuracy']):
        bars=ax.bar(x,test[metric].astype(float),.57,color=C1)
        annotate(ax,bars)
        ax.set_xticks(x,CODES);setup(ax,.85);ax.set_title(title,fontsize=16)
    fig.suptitle('Macro-F1 and Balanced Accuracy',fontsize=19,fontweight='bold')
    save(PLOT_NAMES[1],fig)

    # 03 - 2x2 confusion matrix for E7's four models. Same Blues palette, class labels.
    fig,axes=plt.subplots(2,2,figsize=(16,13))
    vmax=max(int(test.loc[c,k]) for c in CODES for k in ('tn','fp','fn','tp'))
    for ax,(code,model,label) in zip(axes.flat,MODELS):
        row=test.loc[code]
        mat=np.array([[int(row.tn),int(row.fp)],[int(row.fn),int(row.tp)]])
        ax.imshow(mat,cmap='Blues',vmin=0,vmax=vmax)
        ax.set_xticks([0,1],['Predicted DS','Predicted DR'])
        ax.set_yticks([0,1],['Actual DS','Actual DR'])
        ax.set_title(f'{code} · {label}',pad=15,fontsize=15)
        for (i,j),val in np.ndenumerate(mat):
            ax.text(j,i,f'{val}\n'+[['TN','FP'],['FN','TP']][i][j],
                    ha='center',va='center',fontsize=17,fontweight='bold',
                    color='white' if val>vmax*.65 else 'black')
    fig.suptitle('Confusion Matrix',fontsize=18,fontweight='bold')
    fig.subplots_adjust(hspace=.32,wspace=.28,top=.90)
    save(PLOT_NAMES[2],fig)

    # 04 - Sensitivity/Specificity M1--M4 (E7 holdout only).
    fig,ax=plt.subplots(figsize=(16,9))
    a=ax.bar(x-.2,test.sensitivity_dr.astype(float),.38,color=C1,label='Sensitivity DR-TB')
    b=ax.bar(x+.2,test.specificity_ds.astype(float),.38,color=C2,label='Specificity DS-TB')
    annotate(ax,a,fmt='{:.1%}');annotate(ax,b,fmt='{:.1%}')
    ax.set_xticks(x,names);setup(ax,.95);ax.legend(loc='upper left')
    ax.set_title('Sensitivity and Specificity',fontweight='bold',fontsize=17)
    save(PLOT_NAMES[3],fig)

    # 05 - E7 within-experiment OOF vs holdout descriptive result (not paired).
    fig,ax=plt.subplots(figsize=(16,9))
    a=ax.bar(x-.2,oof.auroc.astype(float),.38,color=C1,label='OOF · 1,887 development images')
    b=ax.bar(x+.2,test.auroc.astype(float),.38,color=C2,label='Holdout ensemble · 333 images')
    annotate(ax,a);annotate(ax,b)
    ax.set_xticks(x,names);setup(ax,.95);ax.legend()
    ax.set_title('OOF and Test AUROC',fontweight='bold',fontsize=17)
    save(PLOT_NAMES[4],fig)

    # 06 - mean ± SAMPLE SD from five E7 validation folds (not a CI).
    fig,ax=plt.subplots(figsize=(16,9))
    ax.bar(x,oof.fold_auroc_mean.astype(float),.57,
           yerr=oof.fold_auroc_sd_sample.astype(float),color=C1,edgecolor='white',
           error_kw={'capsize':7,'elinewidth':2})
    ax.set_xticks(x,names);setup(ax,1.00)
    for i,row in enumerate(oof.itertuples()):
        ax.text(i,float(row.fold_auroc_mean)+float(row.fold_auroc_sd_sample)+.012,
                f'{row.fold_auroc_mean:.3f} ± {row.fold_auroc_sd_sample:.3f}',
                ha='center',fontsize=11)
    ax.set_title('AUROC Across Five Folds',fontweight='bold',fontsize=17)
    ax.set_ylabel('AUROC');save(PLOT_NAMES[5],fig)

    # 07 - Primary M4 classification metrics, OOF and test (within E7).
    fig,ax=plt.subplots(figsize=(16,9));ix=np.arange(len(METRICS))
    a=ax.bar(ix-.2,oof.loc['M4',METRICS].astype(float),.38,color=C1,label='M4 · OOF')
    b=ax.bar(ix+.2,test.loc['M4',METRICS].astype(float),.38,color=C2,label='M4 · Holdout')
    annotate(ax,a);annotate(ax,b)
    ax.set_xticks(ix,LABELS,rotation=12,ha='right');setup(ax,.95);ax.legend()
    ax.set_title('Classification Performance',fontweight='bold',fontsize=17)
    save(PLOT_NAMES[6],fig)

    # 08 - E7 M4 test - OOF, percentage points; protocols differ.
    diff=np.array([float(test.loc['M4',m])-float(oof.loc['M4',m]) for m in METRICS])*100
    fig,ax=plt.subplots(figsize=(16,9))
    ax.barh(np.arange(len(METRICS)),diff,color=[GREEN if d>=0 else RED for d in diff])
    ax.invert_yaxis();ax.set_yticks(np.arange(len(METRICS)),LABELS)
    bounds=max(8,float(np.max(np.abs(diff)))+5)
    ax.set_xlim(-bounds,bounds);ax.axvline(0,color='black',linewidth=.9)
    for i,d in enumerate(diff):
        ax.text(d+(.7 if d>=0 else -.7),i,f'{d:+.2f} pp',
                va='center',ha='left' if d>=0 else 'right',fontsize=12,fontweight='bold')
    ax.set_xlabel('Holdout minus OOF (percentage points)')
    ax.set_title('OOF and Holdout Performance Differences',fontsize=17,fontweight='bold')
    save(PLOT_NAMES[7],fig)

    # 09 - E6-like publication table; E7 model metrics only.
    fig,ax=plt.subplots(figsize=(16,9));ax.axis('off')
    heads=['Model','AUROC','Macro-F1','Balanced Acc.','Sens. DR','Spec. DS']
    rows=[]
    for code in CODES:
        r=test.loc[code]
        rows.append([code+' · '+MODEL_NAME[code]]+[f'{float(r[m]):.4f}' for m in METRICS])
    table=ax.table(cellText=rows,colLabels=heads,loc='center',cellLoc='center',
                   colLoc='center',colWidths=[.29,.13,.14,.16,.14,.14])
    table.auto_set_font_size(False);table.set_fontsize(13);table.scale(1,2.7)
    for (i,j),cell in table.get_celld().items():
        cell.set_linewidth(.3)
        if i==0:cell.set_facecolor('#26476B');cell.set_text_props(color='white',weight='bold')
        elif i==4:cell.set_facecolor('#EAF0F7');cell.set_text_props(weight='bold')
        elif i%2==0:cell.set_facecolor('#F5F7FA')
    ax.set_title('Holdout Test Performance',pad=28,fontsize=17,fontweight='bold')
    save(PLOT_NAMES[8],fig)

def write_analysis(test,oof,checks):
    m=test.loc['M4']; v=oof.loc['M4']
    lines=[
      '# Eksperimen 7 — Analisis Visualisasi Akademis','',
      '## Protokol','',
      '- Empat model E7: M1 Frozen RAW, M2 Frozen CLAHE, M3 LoRA RAW, M4 LoRA CLAHE.',
      '- Development: 1.887 citra (patient-level 5-fold OOF), dan holdout 333 citra / 259 pasien.',
      '- Holdout: 166 DS-TB dan 167 DR-TB; lima checkpoint dirata-ratakan probabilitasnya.',
      '- M4 LoRA CLAHE dipilih sebelum holdout testing berdasarkan pooled image-level OOF AUROC.',
      '- Threshold primer: 0,50. Threshold 0,44 hanya analisis eksploratif terpisah.',
      '- Semua hasil merupakan evaluasi per citra; tidak ada agregasi per pasien.','',
      '## Hasil holdout Eksperimen 7','',
      '| Model | AUROC | Macro-F1 | Balanced Acc. | Sens. DR | Spec. DS | TN/FP/FN/TP |',
      '|---|---:|---:|---:|---:|---:|---|',
    ]
    for code in CODES:
        r=test.loc[code]
        lines.append(f'| {code} {MODEL_NAME[code]} | {r.auroc:.6f} | {r.macro_f1:.6f} | '
                     f'{r.balanced_accuracy:.6f} | {r.sensitivity_dr:.6f} | {r.specificity_ds:.6f} | '
                     f'{int(r.tn)}/{int(r.fp)}/{int(r.fn)}/{int(r.tp)} |')
    lines += ['', '## Fokus model utama M4','',
        f'- Pooled OOF AUROC: **{v.auroc:.6f}**; holdout AUROC: **{m.auroc:.6f}**.',
        f'- Holdout Macro-F1: {m.macro_f1:.6f}; balanced accuracy: {m.balanced_accuracy:.6f}.',
        f'- Sensitivitas DR-TB: {m.sensitivity_dr:.6f}; spesifisitas DS-TB: {m.specificity_ds:.6f}.',
        f'- Confusion matrix TN/FP/FN/TP: {int(m.tn)}/{int(m.fp)}/{int(m.fn)}/{int(m.tp)}.',
        '', '## Keterbatasan metodologis','',
        '1. Holdout ini digunakan kembali dalam seri eksperimen, sehingga bersifat eksploratif, bukan validasi eksternal baru.',
        '2. Prediksi OOF development berasal dari checkpoint validasi fold; hasil OOF tidak setara dengan independent nested CV.',
        '3. Perbandingan OOF vs holdout dalam gambar 05, 07, dan 08 bersifat deskriptif; kedua sumber memiliki protokol prediksi berbeda.',
        '4. Simpangan baku AUROC antar-five-fold bukan confidence interval.',
        '5. Tidak ada klaim signifikansi statistik maupun kelayakan klinis.',
        '6. Kurva ROC dihitung dari probabilitas holdout E7 per citra, bukan direkonstruksi dari AUROC agregat.',
        '', '## Jejak integritas sumber',''
    ]
    for key,val in checks.items(): lines.append(f'- {key}: {val}')
    return '\n'.join(lines)+'\n'

def render_table(test,oof):
    rows=[]
    for code in CODES:
        a=test.loc[code];v=oof.loc[code]
        row={'model_code':code,'model':MODEL_NAME[code],'primary_m4':code=='M4'}
        for metric in METRICS:
            row['holdout_'+metric]=float(a[metric]);row['oof_'+metric]=float(v[metric])
        row['fold_auroc_mean']=float(v.fold_auroc_mean)
        row['fold_auroc_sample_sd']=float(v.fold_auroc_sd_sample)
        for metric in ('tn','fp','fn','tp'): row[metric]=int(a[metric])
        rows.append(row)
    return pd.DataFrame(rows)

def output_files():
    return PLOT_NAMES+['13_e7_final_metrics.csv','13_e7_final_analysis.md']

def verify_outputs(out,paths,aux):
    need(out.is_dir(),f'Output tidak ditemukan: {out}')
    report=load_json(out/'13_e7_visualization_audit_report.json')
    need(report.get('status')=='PASS_E7_ONLY_ACADEMIC_VISUALIZATIONS','Status audit salah')
    need(report.get('chart_titles')==TITLE_MAP,'Daftar grafik berubah')
    need(report.get('input_sha256')=={k:sha(v) for k,v in paths.items() if v is not None},
         'Sumber E7 berubah sejak visualisasi')
    need(report.get('training_history_sha256')==aux['history'],'History E7 berubah')
    need(report.get('roc_prediction_sha256')==aux['roc'],'Prediksi holdout E7 berubah')
    need(len(report.get('output_sha256',{}))==len(output_files()),'Jumlah output tidak sesuai')
    for filename,digest in report['output_sha256'].items():
        p=out/filename
        need(p.is_file() and sha(p)==digest,f'Output berubah/hilang: {p}')
    print(f'PASS_E7_ONLY_VISUALIZATIONS_VERIFIED | PNG={len(PLOT_NAMES)} | E7 ONLY')

def run(args):
    root=Path(args.e7_root).expanduser().resolve() if args.e7_root else Path(__file__).resolve().parent.parent
    out=Path(args.output).expanduser().resolve() if args.output else root/'output'/'visualizations'/'13_e7_only_academic'
    paths,locks,checks,test,oof=discover_sources(root)
    oof=gather_fold_statistics(oof,root,paths)
    histories,predictions,aux=load_auxiliary(root,locks['e7_oof_lock'],locks['e7_test_lock'],test)
    if args.check:
        print('PASS_E7_ONLY_SOURCE_AUDIT')
        print('Eksperimen 7 SAJA | OOF M4 %.6f | Holdout M4 %.6f'%(oof.loc['M4','auroc'],test.loc['M4','auroc']))
        print('20 training history dan 4 prediksi holdout telah dibaca; tidak ada training/inferensi.')
        for k,p in paths.items():
            if p: print('SOURCE:',k,p,'|',checks.get(k,'sha256_calculated'))
        print('TARGET:',out)
        return
    if args.verify:
        verify_outputs(out,paths,aux)
        return
    need(args.run,'Pilih --check, --run, atau --verify')
    need(not out.exists(),f'Folder output sudah ada dan tidak ditimpa: {out}. Gunakan --output untuk target baru.')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=out.parent/(out.name+'.building-'+uuid.uuid4().hex[:12])
    need(not stage.exists(),'Folder staging sudah ada')
    stage.mkdir()
    try:
        render_plots(stage,test,oof)
        render_roc(stage,predictions,test)
        render_training_histories(stage,histories)
        for name in PLOT_NAMES:
            file=stage/name
            need(file.is_file() and file.stat().st_size>5000,f'Gambar gagal: {name}')
        render_table(test,oof).to_csv(stage/'13_e7_final_metrics.csv',index=False,float_format='%.12g')
        (stage/'13_e7_final_analysis.md').write_text(write_analysis(test,oof,checks),encoding='utf-8')
        report={
          'status':'PASS_E7_ONLY_ACADEMIC_VISUALIZATIONS','experiment':'experiment7',
          'style_reference':'07_visualisasi_e6_akademis.py',
          'chart_count':len(PLOT_NAMES),'chart_titles':TITLE_MAP,
          'e7_only':True,'other_experiment_input_used':False,
          'primary_model':'M4 — lora_clahe','primary_selected_before_testing':True,
          'primary_threshold':0.5,'alternative_threshold_0_44_not_used_in_primary_plots':True,
          'development_images':1887,'test_images':333,'test_patients':259,
          'test_class_counts':{'DS-TB':166,'DR-TB':167},
          'reused_holdout_exploratory':True,
          'new_training_performed':False,'new_inference_performed':False,'test_pixel_read':False,
          'input_sha256':{k:sha(v) for k,v in paths.items() if v is not None},
          'source_checks':checks,
          'training_history_sha256':aux['history'],
          'roc_prediction_sha256':aux['roc'],
          'auxiliary_checks':aux['integrity'],
          'output_sha256':{name:sha(stage/name) for name in output_files()},
          'caveats':['OOF and holdout use different prediction protocols',
                     'image-level evaluation','reused exploratory holdout',
                     'fold SD is not a confidence interval','no statistical significance claim'],
        }
        (stage/'13_e7_visualization_audit_report.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        stage.rename(out)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'PASS_E7_ONLY_ACADEMIC_VISUALIZATIONS | {len(PLOT_NAMES)} PNG | 0 external-experiment inputs')
    print('OUTPUT:',out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--verify',action='store_true')
    parser.add_argument('--e7-root',help='Folder eksperimen7, jika skrip ditempatkan di luar eksperimen7/src')
    parser.add_argument('--output',help='Folder output baru, tidak akan menimpa yang lama')
    args=parser.parse_args()
    try:run(args)
    except Exception as e:
        print(f'FAIL_E7_ONLY_VISUALIZATION: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
