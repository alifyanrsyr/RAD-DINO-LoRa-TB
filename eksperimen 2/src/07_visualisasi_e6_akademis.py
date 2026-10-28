from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = [('M1','frozen_raw','Frozen RAW'), ('M2','frozen_clahe','Frozen CLAHE'),
          ('M3','lora_raw','LoRA RAW'), ('M4','lora_clahe','LoRA CLAHE')]
CODES = [a for a,_,_ in MODELS]
MODEL_NAME = {a:c for a,b,c in MODELS}
NAME_TO_CODE = {b:a for a,b,c in MODELS}
METRICS = ['auroc','macro_f1','balanced_accuracy','sensitivity_dr','specificity_ds']
LABELS = ['AUROC','Macro-F1','Balanced accuracy','Sensitivity DR-TB','Specificity DS-TB']
PLOT_NAMES = [
    '01_test_auroc_e5_vs_e6.png',
    '02_test_macro_f1_balanced_e5_vs_e6.png',
    '03_primary_confusion_matrices.png',
    '04_primary_sensitivity_specificity.png',
    '05_e6_oof_vs_test_auroc.png',
    '06_e6_fold_auroc_mean_sd.png',
    '07_primary_metrics_e5_vs_e6.png',
    '08_primary_metric_deltas.png',
    '09_e5_e6_final_comparison_table.png',
    '10_roc_curve_e6.png',
] + [f'{11+i*2:02d}_{model}_loss.png' for i,model in enumerate(['frozen_raw','frozen_clahe','lora_raw','lora_clahe'])] + [f'{12+i*2:02d}_{model}_accuracy.png' for i,model in enumerate(['frozen_raw','frozen_clahe','lora_raw','lora_clahe'])]

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def load_json(path):
    need(path.is_file(), f'Missing JSON file: {path}')
    with path.open(encoding='utf-8') as f: return json.load(f)

def close(a,b): return math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-10)

def inputs(e5root,e6root,e5_csv_reference=None):
    e5d=e5root/'output/testing/11_final_holdout_test_v1'
    e6d=e6root/'output/testing/06_e6_five_fold_ensemble_holdout_v1'
    oof=e6root/'output/cross_validation/05_oof_model_selection_v1'
    return {
        'e5_csv':Path(e5_csv_reference).resolve() if e5_csv_reference else e5d/'11_test_comparison.csv',
        'e5_lock':e5d/'11_final_holdout_test_report.json',
        'e5_test_manifest':e5root/'data/manifests/experiment5_split_70_15_15_seed42_v1/test.csv',
        'e6_csv':e6d/'06_e6_ensemble_test_comparison.csv',
        'e6_lock':e6d/'06_e6_ensemble_holdout_test_LOCKED.json',
        'oof_csv':oof/'05_e6_oof_comparison_4models.csv',
        'oof_lock':oof/'05_e6_oof_model_selection_LOCKED.json',
    }

def audited_data(paths):
    for k,p in paths.items(): need(p.is_file(),f'Missing {k}: {p}')
    e5_lock=load_json(paths['e5_lock']); e6_lock=load_json(paths['e6_lock']); oof_lock=load_json(paths['oof_lock'])
    need(e5_lock.get('status')=='PASS_FINAL_HOLDOUT_TEST','E5 test not locked PASS')
    need(e6_lock.get('status')=='PASS_E6_FIXED_HOLDOUT_5FOLD_ENSEMBLE','E6 test not locked PASS')
    need(oof_lock.get('status')=='PASS_E6_OOF_MODEL_SELECTION_LOCKED','E6 OOF not locked PASS')
    need(e5_lock.get('primary_model_selected_before_test',{}).get('model_code')=='M4','E5 primary lock changed')
    need(e6_lock.get('primary_model_code')=='M4' and e6_lock.get('primary_model')=='lora_clahe','E6 primary lock changed')
    need(oof_lock.get('primary_model')=='lora_clahe','E6 OOF primary lock changed')
    need(e6_lock.get('primary_selection_locked_before_testing') is True,'E6 test model selected after test')
    need(e6_lock.get('test_based_model_selection') is False,'E6 test-based selection flagged')
    need(e5_lock.get('test_based_model_selection_performed') is False,'E5 test-based selection flagged')
    need(e6_lock.get('test_set_is_reused_e5_holdout_exploratory') is True,'Reused holdout limitation missing')
    need(e5_lock.get('n_test_images')==e6_lock['test_cohort']['images']==333,'Test image cohort mismatch')
    need(e5_lock.get('n_test_patients')==e6_lock['test_cohort']['patients']==259,'Test patient cohort mismatch')
    need(e5_lock.get('test_class_counts')=={'DS-TB':166,'DR-TB':167}==
         {'DS-TB':e6_lock['test_cohort']['DS-TB'],'DR-TB':e6_lock['test_cohort']['DR-TB']},'Test class counts mismatch')
    need(sha(paths['e5_test_manifest']) == e6_lock['test_manifest_sha256'], 'E5 fixed test manifest changed since E6')
    need(sha(paths['e5_csv']) == e5_lock['output_sha256']['11_test_comparison.csv'],
         'E5 CSV hash mismatch | file='+str(paths['e5_csv'])+
         ' | expected='+e5_lock['output_sha256']['11_test_comparison.csv']+
         ' | actual='+sha(paths['e5_csv']))
    need(sha(paths['e6_csv']) == e6_lock['output_sha256']['06_e6_ensemble_test_comparison.csv'],'E6 CSV hash mismatch')
    need(sha(paths['oof_csv']) == oof_lock['output_sha256']['05_e6_oof_comparison_4models.csv'],'E6 OOF CSV hash mismatch')
    need(sha(paths['oof_lock'])==e6_lock['stage5_model_selection_lock_sha256'],'OOF lock changed after test')
    need(oof_lock['n_development_images']==1887 and oof_lock['n_training_runs']==20,'E6 OOF dataset/training count changed')
    need(close(oof_lock['primary_model_auroc'],e6_lock['primary_model_oof_auroc']),'Primary OOF AUROC changed')
    need(close(e5_lock['decision_threshold'],e6_lock['threshold']) and close(e6_lock['threshold'],0.5),'Threshold mismatch')
    e5=pd.read_csv(paths['e5_csv'],dtype={'model_code':str}).set_index('model_code',drop=False)
    e6=pd.read_csv(paths['e6_csv'],dtype={'model_code':str}).set_index('model_code',drop=False)
    oof=pd.read_csv(paths['oof_csv'],dtype={'model':str});oof['model_code']=oof['model'].map(NAME_TO_CODE)
    need(set(e5.index)==set(e6.index)==set(oof.model_code)==set(CODES),'Exactly four model codes required')
    need(len(e5)==len(e6)==len(oof)==4,'Duplicate model rows')
    oof=oof.set_index('model_code',drop=False)
    for code in CODES:
        a,b,c = e5.loc[code], e6.loc[code], oof.loc[code]
        need(a['method']==b['method'] and a['variant']==b['variant'],'Model identity mismatch '+code)
        need(a['test_images']==b['n_test_images']==333 and a['test_patients']==b['n_test_patients']==259,
             'Test counts wrong '+code)
        need(int(b['n_folds'])==5 and int(c['oof_images'])==1887 and int(c['n_folds'])==5,'Fold count mismatch '+code)
        need(bool(a['preselected_primary_model'])==(code=='M4'),'E5 primary flags wrong '+code)
        need(bool(b['primary_preselected'])==(code=='M4'),'E6 primary flags wrong '+code)
        for row in (a,b):
            tn,fp,fn,tp=(int(row[k]) for k in ('tn','fp','fn','tp'))
            need((tn+fp,fn+tp)==(166,167),'Test confusion counts mismatch '+code)
            need(close(row['accuracy'],(tn+tp)/333), 'Accuracy mismatch '+code)
            need(close(row['sensitivity_dr'],tp/167),'Sensitivity mismatch '+code)
            need(close(row['specificity_ds'],tn/166),'Specificity mismatch '+code)
            need(close(row['balanced_accuracy'],(tp/167+tn/166)/2),'Balanced accuracy mismatch '+code)
            for metric in METRICS: need(0<=float(row[metric])<=1,'Metric out of range '+code)
        for metric in METRICS: need(0<=float(c['oof_'+metric] if ('oof_'+metric) in c.index else c[metric])<=1,
                                   'OOF metric range mismatch '+code)
        need(close(float(a['threshold']),.5) and close(float(b['threshold']),.5),'Threshold mismatch '+code)
        jrow=next(x for x in e6_lock['model_comparison'] if x['model_code']==code)
        for metric in METRICS: need(close(float(b[metric]),jrow[metric]), 'E6 CSV/JSON mismatch '+code+'/'+metric)
        jrow=next(x for x in e5_lock['all_model_test_metrics'] if x['model_code']==code)
        for metric in METRICS: need(close(float(a[metric]),jrow[metric]), 'E5 CSV/JSON mismatch '+code+'/'+metric)
    need(close(float(e6.loc['M4','auroc']),e6_lock['primary_model_test_metrics']['auroc']),'M4 test AUROC mismatch')
    need(close(float(oof.loc['M4','oof_auroc']),oof_lock['primary_model_auroc']),'M4 OOF AUROC mismatch')
    return e5.loc[CODES],e6.loc[CODES],oof.loc[CODES],e5_lock,e6_lock,oof_lock


# Judul gambar tetap ringkas; metodologi diterangkan di legenda, sumbu, dan laporan.
TITLE_MAP = {
    '01_test_auroc_e5_vs_e6.png': 'AUROC Comparison',
    '02_test_macro_f1_balanced_e5_vs_e6.png': 'Macro-F1 and Balanced Accuracy',
    '03_primary_confusion_matrices.png': 'Confusion Matrix',
    '04_primary_sensitivity_specificity.png': 'Sensitivity and Specificity',
    '05_e6_oof_vs_test_auroc.png': 'OOF and Test AUROC',
    '06_e6_fold_auroc_mean_sd.png': 'AUROC Across Five Folds',
    '07_primary_metrics_e5_vs_e6.png': 'Classification Performance',
    '08_primary_metric_deltas.png': 'Performance Differences',
    '09_e5_e6_final_comparison_table.png': 'Performance Comparison of Experiments 5 and 6',
    '10_roc_curve_e6.png': 'Receiver Operating Characteristic (ROC) Curve',
}
for _model in ('frozen_raw','frozen_clahe','lora_raw','lora_clahe'):
    _idx = ('frozen_raw','frozen_clahe','lora_raw','lora_clahe').index(_model)
    TITLE_MAP[f'{11+_idx*2:02d}_{_model}_loss.png'] = 'Training and Validation Loss'
    TITLE_MAP[f'{12+_idx*2:02d}_{_model}_accuracy.png'] = 'Training and Validation Accuracy'

def load_auxiliary(e6root, oof_lock, test_lock, e6):
    """Read only verified epoch histories and E6 ensemble predictions.

    Never loads test images or model checkpoints; ROC uses saved test probabilities.
    """
    history_dir=e6root/'output/training'
    test_dir=e6root/'output/testing/06_e6_five_fold_ensemble_holdout_v1'
    histories={}; digest_hist={}; digest_roc={}; predictions={}
    for code,model,_ in MODELS:
        histories[model]=[]
        for fold in range(1,6):
            run=f'{model}_fold_{fold}'
            p=history_dir/run/'training_history.csv'
            need(p.is_file(),f'Missing training history: {p}')
            expected=oof_lock['run_artifact_sha256'][run]['training_history.csv']
            need(sha(p)==expected,f'Training history changed: {p}')
            digest_hist[run]=sha(p)
            frame=pd.read_csv(p)
            required={'epoch','train_loss','val_loss','train_accuracy','val_accuracy'}
            need(required.issubset(frame.columns),f'Unexpected columns: {p}')
            need(len(frame)>=2 and frame.epoch.is_unique and frame.epoch.iloc[0]==1,
                 f'Invalid epochs: {p}')
            for col in required:
                need(np.isfinite(pd.to_numeric(frame[col],errors='coerce')).all(),
                     f'Invalid {col}: {p}')
            histories[model].append(frame)
        filename=f'06_test_ensemble_predictions_{model}.csv'
        p=test_dir/filename
        need(p.is_file(),f'Missing locked test prediction CSV: {p}')
        expected=test_lock['output_sha256'][filename]
        need(sha(p)==expected,f'Locked prediction hash mismatch: {p}')
        digest_roc[filename]=sha(p)
        df=pd.read_csv(p)
        need(len(df)==333 and {'y_true','p_dr_mean_5fold'}.issubset(df.columns),
             f'Unexpected prediction schema: {p}')
        truth=pd.to_numeric(df.y_true,errors='raise').to_numpy(dtype=int)
        prob=pd.to_numeric(df.p_dr_mean_5fold,errors='raise').to_numpy(dtype=float)
        need(set(truth)=={0,1} and (truth==0).sum()==166 and (truth==1).sum()==167,
             f'Wrong test labels: {p}')
        need(np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),
             f'Invalid probabilities: {p}')
        predictions[code]=(truth,prob)
    return histories,predictions,{'history':digest_hist,'roc':digest_roc}

def render_roc(stage,preds,e6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fig,ax=plt.subplots(figsize=(12,8))
    colors=['#426AA3','#DD9446','#40856D','#95549E']
    for (code,_,label),color in zip(MODELS,colors):
        truth,prob=preds[code]
        score=roc_auc_score(truth,prob)
        need(abs(score-float(e6.loc[code,'auroc']))<1e-9,
             f'ROC AUROC mismatch for {code}')
        fpr,tpr,_=roc_curve(truth,prob)
        ax.plot(fpr,tpr,label=f'{code} — {label} (AUROC {score:.3f})',lw=2,color=color)
    ax.plot([0,1],[0,1],ls='--',lw=1,color='gray')
    ax.set_xlim(0,1);ax.set_ylim(0,1.02)
    ax.set_xlabel('False Positive Rate');ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve',fontsize=17,pad=18)
    ax.legend(loc='lower right',fontsize=10);ax.grid(alpha=.18)
    fig.savefig(stage/'10_roc_curve_e6.png',dpi=140,bbox_inches='tight',pad_inches=.3)
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

def render_plots(stage,e5,e6,oof):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    x=np.arange(4); labels=[MODEL_NAME[z] for z in CODES]
    C1='#436FA6';C2='#D38D48';GREEN='#40856D';RED='#AE5F5F'
    def save(name,fig):
        fig.savefig(stage/name,dpi=120,bbox_inches='tight',pad_inches=.35)
        plt.close(fig)
    def annotate(ax, bars,fmt='{:.3f}'):
        for p in bars:
            ax.text(p.get_x()+p.get_width()/2,p.get_height()+0.006,fmt.format(p.get_height()),
                    ha='center',va='bottom',fontsize=10)
    def pair(metric,title,name,ylim=(0,0.90)):
        fig,ax=plt.subplots(figsize=(16,9))
        bars1=ax.bar(x-.2,e5[metric],.38,label='Eksperimen 5 — single model',color=C1)
        bars2=ax.bar(x+.2,e6[metric],.38,label='Eksperimen 6 — 5-model ensemble',color=C2)
        annotate(ax,bars1);annotate(ax,bars2)
        ax.set_ylim(*ylim);ax.set_xticks(x,labels)
        ax.set_ylabel(metric.replace('_',' ').title());ax.set_title(title,pad=20,fontweight='bold',fontsize=17)
        ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        save(name,fig)
    pair('auroc','AUROC Comparison','01_test_auroc_e5_vs_e6.png')
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    for ax,metric,title in zip(axes,['macro_f1','balanced_accuracy'],['Macro-F1','Balanced Accuracy']):
        ax.bar(x-.2,e5[metric],.38,label='E5',color=C1)
        ax.bar(x+.2,e6[metric],.38,label='E6',color=C2)
        ax.set_xticks(x,['M1','M2','M3','M4']);ax.set_ylim(0,.85);ax.set_title(title,fontsize=16)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    fig.suptitle('Macro-F1 and Balanced Accuracy',fontsize=19,fontweight='bold')
    save('02_test_macro_f1_balanced_e5_vs_e6.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(16,9))
    maxv=max(max(e5.loc['M4',k],e6.loc['M4',k]) for k in ('tn','fp','fn','tp'))
    for ax,df,title in zip(axes,[e5,e6],['Experiment 5','Experiment 6']):
        r=df.loc['M4'];mat=np.array([[int(r['tn']),int(r['fp'])],[int(r['fn']),int(r['tp'])]])
        ax.imshow(mat,cmap='Blues',vmin=0,vmax=maxv)
        ax.set_xticks([0,1],['Predicted DS','Predicted DR']);ax.set_yticks([0,1],['Actual DS','Actual DR'])
        ax.set_title(title,pad=15,fontsize=16)
        for (i,j),val in np.ndenumerate(mat):
            ax.text(j,i,f'{val}\n'+[['TN','FP'],['FN','TP']][i][j],ha='center',va='center',
                    fontsize=20,fontweight='bold',color='white' if val>maxv*.65 else 'black')
    fig.suptitle('Confusion Matrix',fontsize=18,fontweight='bold')
    save('03_primary_confusion_matrices.png',fig)
    fig,ax=plt.subplots(figsize=(16,9))
    idx=np.arange(2); sens=[float(e5.loc['M4','sensitivity_dr']),float(e6.loc['M4','sensitivity_dr'])]
    spec=[float(e5.loc['M4','specificity_ds']),float(e6.loc['M4','specificity_ds'])]
    s=ax.bar(idx-.2,sens,.38,color=C1,label='Sensitivity DR-TB')
    p=ax.bar(idx+.2,spec,.38,color=C2,label='Specificity DS-TB')
    annotate(ax,s,fmt='{:.1%}');annotate(ax,p,fmt='{:.1%}')
    ax.set_ylim(0,.95);ax.set_xticks(idx,['E5 single M4','E6 ensemble M4'])
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Sensitivity and Specificity',fontsize=17,fontweight='bold')
    save('04_primary_sensitivity_specificity.png',fig)
    fig,ax=plt.subplots(figsize=(16,9))
    a=ax.bar(x-.2,oof['oof_auroc'],.38,color=C1,label='E6 OOF AUROC · development 1,887')
    b=ax.bar(x+.2,e6['auroc'],.38,color=C2,label='E6 test AUROC · holdout 333')
    annotate(ax,a);annotate(ax,b)
    ax.set_ylim(0,.9);ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend();ax.set_title('OOF and Test AUROC',fontweight='bold',fontsize=17)
    save('05_e6_oof_vs_test_auroc.png',fig)
    fig,ax=plt.subplots(figsize=(16,9))
    ax.bar(x,oof['fold_auroc_mean'],.57,yerr=oof['fold_auroc_sd_sample'],
           color=C1,edgecolor='white',error_kw={'capsize':7,'elinewidth':2})
    ax.set_xticks(x,labels);ax.set_ylim(0,.9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for i,r in enumerate(oof.itertuples()):
        ax.text(i,float(r.fold_auroc_mean)+float(r.fold_auroc_sd_sample)+.012,
                f'{r.fold_auroc_mean:.3f} ± {r.fold_auroc_sd_sample:.3f}',ha='center',fontsize=11)
    ax.set_title('AUROC Across Five Folds',fontweight='bold',fontsize=17)
    ax.set_ylabel('AUROC');save('06_e6_fold_auroc_mean_sd.png',fig)
    fig,ax=plt.subplots(figsize=(16,9)); ix=np.arange(len(METRICS))
    a=ax.bar(ix-.2,e5.loc['M4',METRICS].astype(float),.38,color=C1,label='E5 · M4')
    b=ax.bar(ix+.2,e6.loc['M4',METRICS].astype(float),.38,color=C2,label='E6 · M4')
    annotate(ax,a);annotate(ax,b);ax.set_xticks(ix,LABELS,rotation=12,ha='right');ax.set_ylim(0,.92)
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Classification Performance',fontweight='bold',fontsize=17)
    save('07_primary_metrics_e5_vs_e6.png',fig)
    diff=np.array([float(e6.loc['M4',m])-float(e5.loc['M4',m]) for m in METRICS])*100
    fig,ax=plt.subplots(figsize=(16,9));bars=ax.barh(np.arange(len(METRICS)),diff,
                     color=[GREEN if d>=0 else RED for d in diff]);ax.invert_yaxis()
    ax.set_yticks(np.arange(len(METRICS)),LABELS);ax.set_xlim(min(-20,min(diff)-5),max(20,max(diff)+5))
    ax.axvline(0,color='black',linewidth=.9)
    for i,d in enumerate(diff):
        ax.text(d+(.7 if d>=0 else -.7),i,f'{d:+.2f} pp',
                va='center',ha='left' if d>=0 else 'right',fontsize=12,fontweight='bold')
    ax.set_xlabel('E6 minus E5 (percentage points)');
    ax.set_title('Performance Differences',fontsize=17,fontweight='bold')
    save('08_primary_metric_deltas.png',fig)
    fig,ax=plt.subplots(figsize=(16,9));ax.axis('off')
    headers=['Model','E5 AUROC','E6 AUROC','Delta AUROC','E5 Macro-F1','E6 Macro-F1']
    rows=[]
    for code in CODES:
        d=float(e6.loc[code,'auroc'])-float(e5.loc[code,'auroc'])
        rows.append([code+' · '+MODEL_NAME[code],f'{e5.loc[code,"auroc"]:.4f}',
                     f'{e6.loc[code,"auroc"]:.4f}',f'{d:+.4f}',
                     f'{e5.loc[code,"macro_f1"]:.4f}',f'{e6.loc[code,"macro_f1"]:.4f}'])
    tab=ax.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center',colLoc='center',
                 colWidths=[.24,.13,.13,.15,.17,.17]);tab.auto_set_font_size(False)
    tab.set_fontsize(13);tab.scale(1,2.7)
    for (i,j),cell in tab.get_celld().items():
        cell.set_linewidth(.3)
        if i==0: cell.set_facecolor('#26476B');cell.set_text_props(color='white',weight='bold')
        elif i==4: cell.set_facecolor('#EAF0F7');cell.set_text_props(weight='bold')
        elif i%2==0: cell.set_facecolor('#F5F7FA')
    ax.set_title('Performance Comparison of Experiments 5 and 6',pad=28,fontsize=17,fontweight='bold')
    save('09_e5_e6_final_comparison_table.png',fig)

def markdown_report(e5,e6,oof, hashes):
    d=e6.loc['M4']; old=e5.loc['M4']; vals={m:float(d[m])-float(old[m]) for m in METRICS}
    lines=[
      '# Eksperimen 6 — Analisis Akhir 5-Fold Ensemble vs Eksperimen 5', '',
      '## Protokol', '',
      '- E5: satu model per konfigurasi, 70% training dan 15% fixed validation.',
      '- E6: 85% development untuk StratifiedGroupKFold lima fold, ensemble rerata probabilitas lima best checkpoint per konfigurasi.',
      '- Testing yang sama: **333 citra / 259 pasien** (166 DS-TB, 167 DR-TB).',
      '- M4 (LoRA + CLAHE) telah dipilih sebelum testing di kedua eksperimen.',
      '- E6 dipilih berdasarkan pooled **image-level OOF AUROC = %.6f**.' % float(oof.loc['M4','oof_auroc']),
      '- Threshold yang dilaporkan: 0,5 (tidak dioptimasi pada testing).', '',
      '## Hasil final testing', '',
      '| Konfigurasi | E5 AUROC | E6 AUROC | Delta | E6 Macro-F1 | E6 Balanced Acc |',
      '|---|---:|---:|---:|---:|---:|',
    ]
    for code in CODES:
        v=e5.loc[code];n=e6.loc[code]
        lines.append(f'| {code} — {MODEL_NAME[code]} | {v.auroc:.6f} | {n.auroc:.6f} | '
                     f'{n.auroc-v.auroc:+.6f} | {n.macro_f1:.6f} | {n.balanced_accuracy:.6f} |')
    lines+=['','## Model utama M4', '',
      f'- AUROC: {old.auroc:.6f} (E5) menjadi **{d.auroc:.6f}** (E6), perubahan {vals["auroc"]:+.6f}.',
      f'- Macro-F1: {old.macro_f1:.6f} menjadi {d.macro_f1:.6f}, perubahan {vals["macro_f1"]:+.6f}.',
      f'- Balanced accuracy: {old.balanced_accuracy:.6f} menjadi {d.balanced_accuracy:.6f}, perubahan {vals["balanced_accuracy"]:+.6f}.',
      f'- DR sensitivity: {old.sensitivity_dr*100:.2f}% menjadi {d.sensitivity_dr*100:.2f}% ({vals["sensitivity_dr"]*100:+.2f} poin persentase).',
      f'- DS specificity: {old.specificity_ds*100:.2f}% menjadi {d.specificity_ds*100:.2f}% ({vals["specificity_ds"]*100:+.2f} poin persentase).',
      f'- E5 M4 confusion matrix TN/FP/FN/TP: {int(old.tn)}/{int(old.fp)}/{int(old.fn)}/{int(old.tp)}.',
      f'- E6 M4 confusion matrix TN/FP/FN/TP: {int(d.tn)}/{int(d.fp)}/{int(d.fn)}/{int(d.tp)}.', '',
      '## Interpretasi dan batasan', '',
      '1. **AUROC M4 meningkat secara deskriptif**, tetapi sensitivitas untuk DR-TB pada threshold 0,5 menurun; FN naik dari 40 menjadi 67 citra.',
      '2. E6 M4 memiliki test AUROC tertinggi di antara empat konfigurasi; M1/M3 unggul pada sebagian metrik threshold-based. Model tidak dipilih ulang dari hasil test.',
      '3. E6 menggunakan holdout yang sama dan **telah dievaluasi sebelumnya pada E5**. Ini perbandingan eksploratif, bukan validasi konfirmatori baru.',
      '4. Evaluasi berbasis citra, **bukan pasien**. Variasi SD lima fold bukan confidence interval, dan OOF dapat optimistis karena checkpoint diseleksi menggunakan validation fold yang sama.',
      '5. Perbedaan E5–E6 menggabungkan perubahan protokol training dan ensemble sehingga tidak dapat dinyatakan sebagai efek murni 5-fold cross-validation.',
      '6. Tanpa analisis berpasangan dan interval ketidakpastian, **jangan menyebut peningkatan AUROC signifikan secara statistik**.',
      '7. Jika target aplikasi memerlukan sensitivity DR-TB yang tinggi, evaluasi threshold alternatif harus dilakukan pada development set saja dan ditetapkan sebelum pengujian independen baru.',
      '', '## Jejak integritas', '',
    ]
    for k,v in hashes.items(): lines.append(f'- `{k}`: `{v}`')
    return '\n'.join(lines)+'\n'

def prepare_audit(stage,e5,e6,oof,paths,aux_hashes,reference_meta=None):
    for k in PLOT_NAMES:
        need((stage/k).is_file() and (stage/k).stat().st_size>5000,f'Plot missing or too small {k}')
    need((stage/'07_e6_final_comparison.csv').is_file(),'Comparison CSV missing')
    need((stage/'07_e6_final_analysis.md').is_file(),'Analysis markdown missing')
    outputs=PLOT_NAMES+['07_e6_final_comparison.csv','07_e6_final_analysis.md']
    audit={
      'status':'PASS_E6_ACADEMIC_VISUALIZATIONS',
      'experiment':'experiment6',
      'stage':'7B',
      'title_style':'Concise academic titles; no technical subtitles',
      'chart_titles':TITLE_MAP,
      'training_history_sha256':aux_hashes['history'],
      'roc_prediction_sha256':aux_hashes['roc'],
      'primary_model':'M4 — lora_clahe',
      'primary_selected_before_testing':True,
      'test_set_reused_e5_holdout_exploratory':True,
      'test_images':333,'test_patients':259,
      'e6_cv_development_images':1887,
      'e6_cv_folds':5,
      'test_pixel_read':False,
      'new_inference_performed':False,
      'new_training_performed':False,
      'selection_based_on_test':False,
      'e5_m4_test_auroc':float(e5.loc['M4','auroc']),
      'e6_m4_test_auroc':float(e6.loc['M4','auroc']),
      'delta_auroc':float(e6.loc['M4','auroc']-e5.loc['M4','auroc']),
      'e5_m4_sensitivity_dr':float(e5.loc['M4','sensitivity_dr']),
      'e6_m4_sensitivity_dr':float(e6.loc['M4','sensitivity_dr']),
      'input_sha256':{k:sha(p) for k,p in paths.items()},
      'e5_reference_recovery':reference_meta,
      'output_sha256':{name:sha(stage/name) for name in outputs},
      'caveats':['image-level metrics','reused exploratory holdout','no test-based model reselection',
                 'no inferential significance claim','cross-validation and ensemble differ from E5'],
    }
    (stage/'07_e6_visualization_audit_report.json').write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return audit

def run(args):
    e6root=Path(args.e6_root).resolve() if args.e6_root else Path(__file__).resolve().parent.parent
    e5root=Path(args.e5_root).resolve() if args.e5_root else e6root.parent/'eksperimen5'
    paths=inputs(e5root,e6root,args.e5_csv_reference)
    # Do not silently substitute data. Warn when using a verified E5 reference
    # while the original E5 comparison file is different.
    reference_meta=None
    if args.e5_csv_reference:
        original=e5root/'output/testing/11_final_holdout_test_v1/11_test_comparison.csv'
        lockpath=e5root/'output/testing/11_final_holdout_test_v1/11_final_holdout_test_report.json'
        expected=load_json(lockpath)['output_sha256']['11_test_comparison.csv']
        original_actual=sha(original) if original.is_file() else 'MISSING'
        print('E5 VERIFIED REFERENCE MODE; original E5 file NOT modified',flush=True)
        print('Original E5 CSV:', original,flush=True)
        print('Original SHA256:', original_actual,flush=True)
        print('Expected SHA256:', expected,flush=True)
        print('Reference E5 CSV:',paths['e5_csv'],flush=True)
        need(paths['e5_csv'].resolve()!=original.resolve(),
             'Reference must be a distinct file, not the original E5 file')
        if original_actual!=expected:
            print('WARNING: Original E5 CSV differs from locked hash; using exact SHA256-verified reference.',flush=True)
        reference_meta={'reference_used':True,'reference_path':str(paths['e5_csv']),
                        'reference_sha256':sha(paths['e5_csv']),
                        'original_e5_csv_path':str(original),
                        'original_e5_csv_sha256_at_run':original_actual,
                        'locked_expected_sha256':expected,
                        'original_e5_file_modified_by_stage7':False}
    e5,e6,oof,e5lock,e6lock,ooflock=audited_data(paths)
    histories, ensemble_predictions, aux_hashes = load_auxiliary(e6root, ooflock, e6lock, e6)
    out=Path(args.output).resolve() if args.output else e6root/'output/visualizations/07_final_comparison_academic_v2'
    if args.check:
        print('PASS_E6_ACADEMIC_SOURCE_AUDIT',flush=True)
        print('Primary M4 E5 AUROC %.6f -> E6 AUROC %.6f; delta %+.6f'%(
            e5.loc['M4','auroc'],e6.loc['M4','auroc'],e6.loc['M4','auroc']-e5.loc['M4','auroc']))
        print('20 histories and 4 ensemble prediction files checked (SHA256).')
        print('No training/inference/image-pixel read performed. Output:',out)
        return
    if args.verify:
        report=load_json(out/'07_e6_visualization_audit_report.json')
        need(report['status']=='PASS_E6_ACADEMIC_VISUALIZATIONS','Invalid stage7 report status')
        need(report['input_sha256']=={k:sha(p) for k,p in paths.items()},'Source changed after stage7')
        for name,digest in report['output_sha256'].items():
            path=out/name
            need(path.is_file() and sha(path)==digest,'Output modified/missing: '+name)
        need(len(report['output_sha256'])==len(PLOT_NAMES)+2,'Unexpected number of outputs')
        need(report.get('training_history_sha256')==aux_hashes['history'],'Training histories changed')
        need(report.get('roc_prediction_sha256')==aux_hashes['roc'],'ROC predictions changed')
        need(report.get('chart_titles')==TITLE_MAP,'Chart title standard changed')
        print(f'PASS_E6_ACADEMIC_VISUALIZATIONS_VERIFIED | images={len(PLOT_NAMES)} | primary=M4')
        return
    need(args.run,'Use exactly one --check, --run, or --verify')
    need(not out.exists(),'Output already exists; do not overwrite: '+str(out))
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.parent/(out.name+'.building-'+uuid.uuid4().hex[:12])
    need(not tmp.exists(),'Staging directory already exists')
    tmp.mkdir()
    try:
        render_plots(tmp,e5,e6,oof)
        render_roc(tmp,ensemble_predictions,e6)
        render_training_histories(tmp,histories)
        combined=[]
        for code in CODES:
            row={'model_code':code,'name':MODEL_NAME[code], 'primary':code=='M4'}
            for metric in METRICS:
                row['e5_'+metric]=float(e5.loc[code,metric]); row['e6_'+metric]=float(e6.loc[code,metric]);
                row['delta_'+metric]=row['e6_'+metric]-row['e5_'+metric]
            row['e6_oof_auroc']=float(oof.loc[code,'oof_auroc'])
            row['e6_cv_fold_auroc_mean']=float(oof.loc[code,'fold_auroc_mean'])
            row['e6_cv_fold_auroc_sample_sd']=float(oof.loc[code,'fold_auroc_sd_sample'])
            combined.append(row)
        pd.DataFrame(combined).to_csv(tmp/'07_e6_final_comparison.csv',index=False,float_format='%.12g')
        sourcehash={k:sha(p) for k,p in paths.items()}
        (tmp/'07_e6_final_analysis.md').write_text(markdown_report(e5,e6,oof,sourcehash),encoding='utf-8')
        prepare_audit(tmp,e5,e6,oof,paths,aux_hashes,reference_meta)
        tmp.rename(out)
    except Exception:
        shutil.rmtree(tmp,ignore_errors=True)
        raise
    print(f'PASS_E6_ACADEMIC_VISUALIZATIONS | {len(PLOT_NAMES)} PNG + final tables/report')
    print('Output:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    x=p.add_mutually_exclusive_group(required=True)
    x.add_argument('--check',action='store_true')
    x.add_argument('--run',action='store_true')
    x.add_argument('--verify',action='store_true')
    p.add_argument('--e5-root',help='Override E5 root folder (for tests)')
    p.add_argument('--e5-csv-reference',help='Use original SHA256-verified E5 CSV COPY from E6 folder; never modifies E5')
    p.add_argument('--e6-root',help='Override E6 root folder (for tests)')
    p.add_argument('--output',help='Override output folder (for tests)')
    a=p.parse_args()
    try: run(a)
    except Exception as e:
        print(f'FAIL_E6_STAGE7: {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
