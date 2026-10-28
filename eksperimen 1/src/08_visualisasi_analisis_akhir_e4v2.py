from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent if SCRIPT.parent.name.lower() == "src" else SCRIPT.parent
CV = ROOT / "output" / "cross_validation" / "04_oof_model_selection_v1"
TEST = ROOT / "output" / "testing" / "05_e4v2_five_fold_ensemble_holdout_v1"
TRAIN = ROOT / "output" / "training"
OUT = ROOT / "output" / "visualizations" / "08_final_analysis_academic_v1"

MODELS = ["M1", "M2", "M3", "M4"]
NAMES = {"M1": "Frozen RAW", "M2": "Frozen CLAHE", "M3": "LoRA RAW", "M4": "LoRA CLAHE"}
STEMS = {"M1": "frozen_raw", "M2": "frozen_clahe", "M3": "lora_raw", "M4": "lora_clahe"}
INVERSE = {v:k for k,v in STEMS.items()}
COLOR = "#336789"
COLOR_ALT = "#B8854A"
COLOR_ACCENT = "#4C8B7A"
TEXT = "#273340"
FOLDCOLORS = ["#346A91", "#BF8744", "#428A71", "#945C96", "#B45B68"]
METRICS = ["auroc", "macro_f1", "balanced_accuracy", "sensitivity_dr", "specificity_ds"]
METRIC_LABELS = ["AUROC", "Macro-F1", "Balanced Acc.", "Sensitivity", "Specificity"]
EXPECTED_PNG = [
    "00_Performance_Summary.png", "01_Dataset_Composition.png",
    "02_OOF_AUROC.png", "03_Cross_Validation_AUROC.png",
    "04_OOF_Performance.png", "05_Validation_Sensitivity_and_Specificity.png",
    "06_Test_AUROC.png", "07_Test_Performance.png",
    "08_Sensitivity_and_Specificity.png", "09_Classification_Errors.png",
    "10_Confusion_Matrix.png", "11_OOF_and_Test_AUROC.png",
    "12_LoRA_CLAHE_Performance.png", "13_AUROC_Across_Five_Folds.png",
    "14_Training_and_Validation_Loss.png", "15_Training_and_Validation_Accuracy.png",
    "16_Validation_AUROC.png", "17_Best_Epoch.png",
    "18_Generalization_Gap.png", "19_Post_Best_Changes.png"
]


def require(test, message):
    if not bool(test):
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"File tidak ditemukan: {path}")
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    require(actual == expected,
            f"SHA256 mismatch: {path}\nExpected: {expected}\nActual: {actual}")


def read_json(path):
    require(path.is_file(), f"File JSON tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    require(path.is_file(), f"File CSV tidak ditemukan: {path}")
    return pd.read_csv(path)


def close(a,b,tol=1e-8):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def confusion_metrics(row, n_ds, n_dr, context):
    tn,fp,fn,tp = [int(row[k]) for k in ("tn", "fp", "fn", "tp")]
    require((tn+fp,fn+tp)==(n_ds,n_dr), f"Jumlah confusion matrix salah: {context}")
    n=n_ds+n_dr
    sensitivity=tp/n_dr
    specificity=tn/n_ds
    macro = (2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fn+fp))/2
    values = {
        "sensitivity_dr": sensitivity,
        "specificity_ds": specificity,
        "balanced_accuracy": (sensitivity+specificity)/2,
        "accuracy": (tp+tn)/n,
        "macro_f1": macro,
    }
    for col,score in values.items():
        key = f"oof_{col}" if context.startswith("OOF") else col
        require(key in row and close(row[key],score,1e-6),
                f"Metrik {key} tidak konsisten dengan confusion matrix ({context})")
    return tn,fp,fn,tp



def source_data():
    oof_csv=CV / "04_e4v2_oof_comparison_4models.csv"
    oof_json=CV / "04_e4v2_oof_model_selection_LOCKED.json"
    test_csv=TEST / "05_e4v2_ensemble_test_comparison.csv"
    test_json=TEST / "05_e4v2_ensemble_holdout_test_LOCKED.json"

    a=read_json(oof_json)
    b=read_json(test_json)
    require(a.get("status")=="PASS_E4V2_OOF_MODEL_SELECTION_LOCKED",
            "Tahap OOF E4v2 belum terkunci")
    require(b.get("status")=="PASS_E4V2_FIXED_HOLDOUT_5FOLD_ENSEMBLE",
            "Tahap testing E4v2 belum terkunci")
    require(a.get("primary_model")==b.get("primary_model")=="lora_clahe",
            "Model utama tidak konsisten")
    require(a.get("primary_selected_before_e4v2_test") is True and
            b.get("primary_selection_locked_before_testing") is True and
            b.get("test_based_model_selection") is False,
            "Pemilihan model tidak terkunci sebelum testing")
    require(a.get("n_training_runs")==20 and a.get("n_models")==4 and
            a.get("n_development_images")==1411 and
            a.get("n_development_patients")==1091,
            "Jumlah data development/CV tidak sesuai")
    require(b.get("test_cohort",{}).get("images")==257 and
            b["test_cohort"].get("patients")==193 and
            b["test_cohort"].get("DS-TB")==81 and
            b["test_cohort"].get("DR-TB")==176,
            "Jumlah data testing tidak sesuai dengan laporan terkunci")
    require(close(b.get("threshold"), .5), "Threshold E4v2 bukan 0.5")
    require(sha256(oof_json)==b.get("stage4_model_selection_lock_sha256"),
            "Laporan seleksi OOF berubah setelah testing")
    verify_hash(oof_csv,a["output_sha256"][oof_csv.name])
    verify_hash(test_csv,b["output_sha256"][test_csv.name])
    oof=read_csv(oof_csv)
    test=read_csv(test_csv)
    require(len(oof)==len(test)==4,"Harus ada empat konfigurasi model")
    oof["model_code"]=oof.model.map(INVERSE)
    require(oof.model_code.notna().all() and set(oof.model_code)==set(MODELS),
            "Nama model OOF tidak dikenal")
    require(set(test.model_code)==set(MODELS),"Kode model testing tidak lengkap")
    require(oof.model_code.is_unique and test.model_code.is_unique,
            "Duplikat model terdeteksi")
    oof=oof.set_index("model_code").loc[MODELS]
    test=test.set_index("model_code").loc[MODELS]

    for code in MODELS:
        q=oof.loc[code]; t=test.loc[code]
        require(q["model"]==t["model"]==STEMS[code],
                f"Nama model berubah: {code}")
        require(int(q.oof_images)==1411 and int(q.oof_patients)==1091 and
                int(q.n_folds)==5, f"Jumlah OOF salah: {code}")
        require(int(t.n_test_images)==257 and int(t.n_test_patients)==193 and
                int(t.n_folds)==5, f"Jumlah testing salah: {code}")
        require(close(q.threshold,.5) and close(t.threshold,.5),
                f"Threshold salah: {code}")
        confusion_metrics(q,478,933,f"OOF {code}")
        confusion_metrics(t,81,176,f"TEST {code}")
        for v in [q.oof_auroc,q.fold_auroc_mean,q.fold_auroc_sd_sample,t.auroc]:
            require(np.isfinite(v) and 0<=v<=1, f"AUROC tidak valid: {code}")
        found=next((x for x in b["model_comparison"] if x["model_code"]==code),None)
        found_oof=next((x for x in a["models"] if x["model"]==STEMS[code]),None)
        require(found is not None and found_oof is not None,
                "Model tidak ada pada JSON: "+code)
        for key in ["auroc","macro_f1","balanced_accuracy", "accuracy",
                    "sensitivity_dr","specificity_ds"]:
            require(close(t[key],found[key]),
                    f"Metrik testing CSV/JSON berbeda: {code}/{key}")
        for key in ["oof_auroc","oof_macro_f1","oof_balanced_accuracy",
                    "oof_sensitivity_dr","oof_specificity_ds","fold_auroc_mean",
                    "fold_auroc_sd_sample"]:
            require(close(q[key],found_oof[key]),
                    f"Metrik OOF CSV/JSON berbeda: {code}/{key}")
        require(bool(t.primary_preselected)==(code=="M4"),
                "Flag model utama berubah")
    require(close(oof.loc["M4","oof_auroc"],a["primary_model_oof_auroc"]),
            "OOF M4 tidak sesuai laporan")
    require(close(test.loc["M4","auroc"],b["primary_model_test_metrics"]["auroc"]),
            "AUROC testing M4 tidak sesuai laporan")

    return oof,test,a,b,{
        "oof_csv":str(oof_csv),"oof_csv_sha256":sha256(oof_csv),
        "oof_lock":str(oof_json),"oof_lock_sha256":sha256(oof_json),
        "test_csv":str(test_csv),"test_csv_sha256":sha256(test_csv),
        "test_lock":str(test_json),"test_lock_sha256":sha256(test_json),
    }


def histories_data(oof,lock):
    histories={}; records=[]; provenance={}
    hashes=lock.get("run_artifact_sha256",{})
    for m in MODELS:
        for f in range(1,6):
            run=f"{STEMS[m]}_fold_{f}"
            folder=TRAIN / run
            histfile=folder / "training_history.csv"
            summaryfile=folder / "run_summary.json"
            require(run in hashes, f"Run tidak ada di OOF lock: {run}")
            verify_hash(histfile,hashes[run]["training_history.csv"])
            verify_hash(summaryfile,hashes[run]["run_summary.json"])
            provenance[str(histfile)]=sha256(histfile)
            provenance[str(summaryfile)]=sha256(summaryfile)
            s=read_json(summaryfile)
            require(s.get("status")=="PASS_E4V2_FOLD_TRAINING_COMPLETE" and
                    int(s.get("fold"))==f, f"Training belum lengkap: {run}")
            h=read_csv(histfile).sort_values("epoch")
            required=["epoch","train_loss","train_accuracy","val_loss","val_accuracy","val_auroc"]
            require(set(required).issubset(h.columns),f"Kolom riwayat kurang: {run}")
            for col in required:
                numeric=pd.to_numeric(h[col],errors="coerce")
                require(numeric.notna().all() and np.isfinite(numeric).all(),
                        f"Nilai tak hingga/NaN pada {run}: {col}")
            require(h.epoch.astype(int).tolist()==list(range(1,int(s["last_epoch"])+1)),
                    f"Urutan epoch tidak lengkap: {run}")
            best_epoch=int(s["best_epoch"])
            require(1<=best_epoch<=int(s["last_epoch"]),f"Best epoch tidak valid: {run}")
            best=h.loc[h.epoch==best_epoch].iloc[0]
            last=h.iloc[-1]
            require(close(best.val_auroc,s["best_validation_auroc"],1e-6),
                    f"Best AUROC checkpoint tidak sesuai history: {run}")
            if len(h):
                require(h.train_accuracy.between(0,1).all() and h.val_accuracy.between(0,1).all(),
                        f"Accuracy di luar rentang [0,1]: {run}")
                require(h.val_auroc.between(0,1).all(),f"AUROC di luar rentang [0,1]: {run}")
            histories[(m,f)]=h
            records.append({
                "model_code":m,"model":STEMS[m],"fold":f,
                "best_epoch":best_epoch,"last_epoch":int(s["last_epoch"]),
                "best_val_auroc":float(s["best_validation_auroc"]),
                "train_loss_best":float(best.train_loss),"val_loss_best":float(best.val_loss),
                "train_accuracy_best":float(best.train_accuracy),
                "val_accuracy_best":float(best.val_accuracy),
                "accuracy_gap_best":float(best.train_accuracy-best.val_accuracy),
                "train_loss_change_best_to_last":float(last.train_loss-best.train_loss),
                "val_loss_change_best_to_last":float(last.val_loss-best.val_loss),
                "val_auroc_change_best_to_last":float(last.val_auroc-best.val_auroc),
            })
    require(len(histories)==20,"Riwayat training harus 20")
    records=pd.DataFrame(records)
    for m in MODELS:
        q=oof.loc[m]
        rows=records[records.model_code==m].sort_values("fold")
        require(close(rows.best_val_auroc.mean(),q.fold_auroc_mean,1e-6),
                f"Mean AUROC fold tidak konsisten: {m}")
        require(close(rows.best_val_auroc.std(ddof=1),q.fold_auroc_sd_sample,1e-6),
                f"SD AUROC fold tidak konsisten: {m}")
    return histories,records,provenance


def style():
    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":10,"axes.titlesize":13,
        "axes.titleweight":"semibold","axes.labelsize":10,
        "axes.facecolor":"white","figure.facecolor":"white",
        "axes.spines.top":False,"axes.spines.right":False,
        "savefig.facecolor":"white", "grid.alpha":.17,
        "xtick.labelsize":9,"ytick.labelsize":9,
    })


def save(fig,out,name):
    filepath=out/(name+".png")
    fig.savefig(filepath,dpi=220,bbox_inches="tight",pad_inches=.22)
    plt.close(fig)
    return filepath.name


def label_bars(ax, bars, fmt="{:.3f}", offset=4):
    for rect in bars:
        v=rect.get_height()
        ax.annotate(fmt.format(v),(rect.get_x()+rect.get_width()/2,v),
                    xytext=(0,offset),textcoords="offset points",ha="center",va="bottom",fontsize=9)


def bars(ax,values,title,labels=None,color=COLOR,limit=1,fmt="{:.4f}"):
    x=np.arange(len(values)); bar=ax.bar(x,values,width=.58,color=color)
    ax.set_title(title);ax.set_xticks(x,labels if labels is not None else MODELS)
    ax.set_ylim(0,limit);ax.grid(axis="y")
    label_bars(ax,bar,fmt)
    return bar


def pair_bars(ax,xlabels,a,b,title,la="Sensitivity DR-TB",lb="Specificity DS-TB",ymax=1):
    x=np.arange(len(xlabels))
    p=ax.bar(x-.19,a,width=.36,color=COLOR_ALT,label=la)
    q=ax.bar(x+.19,b,width=.36,color=COLOR,label=lb)
    ax.set_ylim(0,ymax);ax.set_xticks(x,xlabels)
    ax.set_title(title);ax.grid(axis="y");ax.legend(frameon=False,ncol=2,fontsize=8)
    label_bars(ax,p,"{:.3f}");label_bars(ax,q,"{:.3f}")


def cm(ax,row,title):
    mat=np.array([[int(row.tn),int(row.fp)],[int(row.fn),int(row.tp)]])
    ax.imshow(mat,cmap="Blues",vmin=0,vmax=max(1,mat.max()))
    for (i,j),v in np.ndenumerate(mat):
        ax.text(j,i,f"{v}\n{[['TN','FP'],['FN','TP']][i][j]}",ha="center",va="center",
                fontsize=11,fontweight="semibold",color="white" if v>mat.max()*.64 else TEXT)
    ax.set_xticks([0,1],["DS-TB","DR-TB"])
    ax.set_yticks([0,1],["DS-TB","DR-TB"])
    ax.set_xlabel("Predicted label");ax.set_ylabel("True label")
    ax.set_title(title,fontsize=11)


def charts(oof,test,hist,records,out):
    style();made=[]
    o=oof.loc[MODELS];t=test.loc[MODELS]
    model_short=MODELS
    full=[NAMES[m] for m in MODELS]
    x=np.arange(4)
    def keep(fig,name): made.append(save(fig,out,name))

    # 00: Overview, all four panels from actual OOF and test reports.
    fig,ax=plt.subplots(2,2,figsize=(12.8,8.1),layout="constrained")
    for axis,values,title in [
        (ax[0,0],o.oof_auroc,"OOF AUROC"),
        (ax[0,1],t.auroc,"Test AUROC"),
        (ax[1,0],t.macro_f1,"Macro-F1"),
        (ax[1,1],t.balanced_accuracy,"Balanced Accuracy")]:
        bars(axis,values,title,color=[COLOR,COLOR,COLOR,COLOR_ALT],limit=.9)
    fig.suptitle("Performance Summary",fontsize=16)
    keep(fig,"00_Performance_Summary")

    # 01: cohort counts, derived from checked confusion matrices.
    dev_ds=int(o.loc["M1","tn"]+o.loc["M1","fp"])
    dev_dr=int(o.loc["M1","fn"]+o.loc["M1","tp"])
    test_ds=int(t.loc["M1","tn"]+t.loc["M1","fp"])
    test_dr=int(t.loc["M1","fn"]+t.loc["M1","tp"])
    ds=np.array([dev_ds,test_ds,dev_ds+test_ds]); dr=np.array([dev_dr,test_dr,dev_dr+test_dr])
    fig,axis=plt.subplots(figsize=(8.5,4.8))
    axis.bar(np.arange(3),ds,color=COLOR,label="DS-TB")
    axis.bar(np.arange(3),dr,bottom=ds,color=COLOR_ALT,label="DR-TB")
    axis.set_title("Dataset Composition");axis.set_xticks(np.arange(3),["Development","Test","Total"])
    axis.set_ylabel("Number of images");axis.set_ylim(0,int((ds+dr).max()*1.17))
    axis.grid(axis="y");axis.legend(frameon=False,ncol=2)
    for i in range(3):
        axis.text(i,ds[i]+dr[i]+20,str(int(ds[i]+dr[i])),ha="center",fontsize=10)
        axis.text(i,ds[i]/2,str(int(ds[i])),ha="center",va="center",color="white",fontsize=9)
        axis.text(i,ds[i]+dr[i]/2,str(int(dr[i])),ha="center",va="center",color="white",fontsize=9)
    keep(fig,"01_Dataset_Composition")

    # 02: pooled out-of-fold AUROC.
    fig,axis=plt.subplots(figsize=(8.8,4.8))
    bars(axis,o.oof_auroc,"OOF AUROC",full,color=[COLOR]*3+[COLOR_ALT],limit=.88)
    axis.set_ylabel("AUROC");keep(fig,"02_OOF_AUROC")

    # 03: sample SD is fold variability, NOT a confidence interval.
    fig,axis=plt.subplots(figsize=(9.2,5))
    values=o.fold_auroc_mean.to_numpy(dtype=float)
    sd=o.fold_auroc_sd_sample.to_numpy(dtype=float)
    b=axis.bar(x,values,color=COLOR,width=.58,yerr=sd,
        error_kw={"capsize":5,"elinewidth":1.3,"ecolor":TEXT})
    for rect,v,err in zip(b,values,sd):
        axis.text(rect.get_x()+rect.get_width()/2,v+err+.015,f"{v:.4f} ± {err:.4f}",
                  ha="center",fontsize=9)
    axis.set_xticks(x,full);axis.set_ylim(0,.88)
    axis.set_title("Cross-Validation AUROC");axis.set_ylabel("AUROC");axis.grid(axis="y")
    keep(fig,"03_Cross_Validation_AUROC")

    # 04: OOF Macro-F1 and Balanced Accuracy.
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    bars(axes[0],o.oof_macro_f1,"Macro-F1",model_short,limit=.85)
    bars(axes[1],o.oof_balanced_accuracy,"Balanced Accuracy",model_short,limit=.85)
    keep(fig,"04_OOF_Performance")

    # 05: OOF DR sensitivity and DS specificity.
    fig,axis=plt.subplots(figsize=(9.4,5))
    pair_bars(axis,full,o.oof_sensitivity_dr,o.oof_specificity_ds,
              "Sensitivity and Specificity")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.0%}"))
    keep(fig,"05_Validation_Sensitivity_and_Specificity")

    # 06-10: holdout test exclusively, SAME 257 images for M1-M4.
    fig,axis=plt.subplots(figsize=(8.8,4.8))
    bars(axis,t.auroc,"Test AUROC",full,color=[COLOR]*3+[COLOR_ALT],limit=.9)
    axis.set_ylabel("AUROC");keep(fig,"06_Test_AUROC")

    fig,axes=plt.subplots(1,2,figsize=(13,5),layout="constrained")
    bars(axes[0],t.macro_f1,"Macro-F1",model_short,limit=.86)
    bars(axes[1],t.balanced_accuracy,"Balanced Accuracy",model_short,limit=.86)
    keep(fig,"07_Test_Performance")

    fig,axis=plt.subplots(figsize=(9.4,5))
    pair_bars(axis,full,t.sensitivity_dr,t.specificity_ds,
              "Sensitivity and Specificity")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f"{v:.0%}"))
    keep(fig,"08_Sensitivity_and_Specificity")

    fig,axes=plt.subplots(1,2,figsize=(13,4.9),layout="constrained")
    bars(axes[0],t.fn,"False Negatives",model_short,color=COLOR_ALT,
         limit=max(10,int(t.fn.max())*1.3),fmt="{:.0f}")
    bars(axes[1],t.fp,"False Positives",model_short,
         limit=max(10,int(t.fp.max())*1.3),fmt="{:.0f}")
    axes[0].set_ylabel("Images");axes[1].set_ylabel("Images")
    keep(fig,"09_Classification_Errors")

    fig,axes=plt.subplots(2,2,figsize=(9.8,9.4),layout="constrained")
    for ax0,m in zip(axes.flat,MODELS): cm(ax0,t.loc[m],f"{m} — {NAMES[m]}")
    fig.suptitle("Confusion Matrix",fontsize=16)
    keep(fig,"10_Confusion_Matrix")

    # 11: clearly label distinct evaluation populations and ensemble operation.
    fig,axis=plt.subplots(figsize=(9.8,5.2))
    pair_bars(axis,model_short,o.oof_auroc,t.auroc,"OOF and Test AUROC",
              "OOF: single fold","Test: 5-model ensemble",ymax=.94)
    fig.text(.5,.005,"OOF (1,411 development images) and ensemble test (257 images) are distinct evaluations.",
             ha="center",color=TEXT,fontsize=9)
    keep(fig,"11_OOF_and_Test_AUROC")

    # 12: primary M4 on TEST ONLY.
    fig,axis=plt.subplots(figsize=(8.8,4.9))
    r=t.loc["M4"]
    vals=[float(r[k]) for k in METRICS]
    bars(axis,vals,"LoRA CLAHE Performance",METRIC_LABELS,color=COLOR_ALT,limit=.94)
    keep(fig,"12_LoRA_CLAHE_Performance")

    # 13: fold-level AUROC, best checkpoint selected by own fold validation.
    fig,axes=plt.subplots(2,2,figsize=(13.1,8.3),layout="constrained")
    for axis,m in zip(axes.flat,MODELS):
        vals=records[records.model_code==m].sort_values("fold").best_val_auroc.to_numpy()
        f=np.arange(1,6)
        axis.plot(f,vals,color=COLOR,marker="o",lw=2)
        axis.set_xticks(f);axis.set_ylim(.50,.84)
        axis.set_xlabel("Fold");axis.set_ylabel("Validation AUROC")
        axis.set_title(NAMES[m]);axis.grid(True)
        for k,v in zip(f,vals): axis.annotate(f"{v:.4f}",(k,v),xytext=(0,7),
                                             textcoords="offset points",ha="center",fontsize=8)
    fig.suptitle("AUROC Across Five Folds",fontsize=16)
    keep(fig,"13_AUROC_Across_Five_Folds")

    # 14-15: every epoch is read from the exact 20 checksummed history files.
    for metric,tr,val,name,title in [
        ("Loss","train_loss","val_loss","14_Training_and_Validation_Loss","Training and Validation Loss"),
        ("Accuracy","train_accuracy","val_accuracy","15_Training_and_Validation_Accuracy","Training and Validation Accuracy"),
    ]:
        fig,axes=plt.subplots(2,2,figsize=(14,9),layout="constrained")
        for axis,m in zip(axes.flat,MODELS):
            for fold,col in enumerate(FOLDCOLORS,1):
                h=hist[(m,fold)]
                axis.plot(h.epoch,h[tr],color=col,lw=1.6,label=f"F{fold} Train")
                axis.plot(h.epoch,h[val],color=col,lw=1.6,ls="--",label=f"F{fold} Val")
            axis.set_title(NAMES[m]);axis.set_xlabel("Epoch")
            axis.set_ylabel(metric);axis.grid(True)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
            if metric=="Accuracy":axis.set_ylim(0,1.03)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=10,frameon=False,
                   bbox_to_anchor=(.5,-.004),fontsize=8)
        fig.suptitle(title,fontsize=16)
        keep(fig,name)

    # 16: validation AUROC by epoch.
    fig,axes=plt.subplots(2,2,figsize=(13.5,8.6),layout="constrained")
    for axis,m in zip(axes.flat,MODELS):
        for fold,col in enumerate(FOLDCOLORS,1):
            h=hist[(m,fold)]
            axis.plot(h.epoch,h.val_auroc,color=col,lw=1.7,label=f"Fold {fold}")
        axis.set_title(NAMES[m]);axis.set_xlabel("Epoch");axis.set_ylabel("Validation AUROC")
        axis.set_ylim(.45,.90);axis.grid(True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=8))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,-.004))
    fig.suptitle("Validation AUROC",fontsize=16)
    keep(fig,"16_Validation_AUROC")

    order=records.set_index(["model_code","fold"]).loc[
        [(m,f) for m in MODELS for f in range(1,6)]].reset_index()
    x20=np.arange(20)
    labels20=[f"{m}\nF{f}" for m in MODELS for f in range(1,6)]
    colors20=[COLOR if i<10 else COLOR_ALT for i in range(20)]

    # 17: best checkpoint epoch per run.
    fig,axis=plt.subplots(figsize=(12,5.1))
    p=axis.bar(x20,order.best_epoch,color=colors20,width=.72)
    axis.set_xticks(x20,labels20,fontsize=8)
    axis.set_ylim(0,max(order.best_epoch)*1.23+1)
    axis.set_ylabel("Epoch");axis.set_title("Best Epoch");axis.grid(axis="y")
    label_bars(axis,p,"{:.0f}",offset=3)
    keep(fig,"17_Best_Epoch")

    # 18: Train accuracy - validation accuracy at the chosen epoch.
    fig,axis=plt.subplots(figsize=(12,5.1))
    values=order.accuracy_gap_best.to_numpy(float)*100
    p=axis.bar(x20,values,color=colors20,width=.72)
    axis.set_xticks(x20,labels20,fontsize=8)
    axis.set_title("Generalization Gap");axis.set_ylabel("Accuracy gap (percentage points)")
    axis.axhline(0,color=TEXT,lw=1)
    axis.set_ylim(min(-6,values.min()-5),max(6,values.max()+5));axis.grid(axis="y")
    for r,v in zip(p,values):
        axis.annotate(f"{v:+.1f}",(r.get_x()+r.get_width()/2,v),
                      xytext=(0,4 if v>=0 else -11),textcoords="offset points",
                      fontsize=7,ha="center")
    keep(fig,"18_Generalization_Gap")

    # 19: Last epoch minus selected best epoch (metric change, not a statistical test).
    fig,axes=plt.subplots(1,2,figsize=(14.5,5.2),layout="constrained")
    for axis,key,title in [
        (axes[0],"val_loss_change_best_to_last","Validation Loss Change"),
        (axes[1],"val_auroc_change_best_to_last","Validation AUROC Change")]:
        v=order[key].to_numpy(float)
        pp=axis.bar(x20,v,width=.70,color=colors20)
        axis.set_xticks(x20,labels20,fontsize=8)
        axis.set_title(title);axis.set_ylabel("Last epoch − best epoch")
        axis.axhline(0,color=TEXT,lw=1);axis.grid(axis="y")
        for r,z in zip(pp,v):
            axis.annotate(f"{z:+.3f}",(r.get_x()+r.get_width()/2,z),
                          xytext=(0,4 if z>=0 else -11),textcoords="offset points",
                          fontsize=7,ha="center")
        lo=min(v.min(),0);hi=max(v.max(),0);margin=max(.008,(hi-lo)*.23)
        axis.set_ylim(lo-margin,hi+margin)
    fig.suptitle("Post-Best Changes",fontsize=16)
    keep(fig,"19_Post_Best_Changes")

    require(made==EXPECTED_PNG, "Jumlah/urutan grafik tidak cocok dengan rencana 20 gambar")
    return made



def analysis_markdown(oof,test,records,figs):
    lines=[
        "# Analisis Akhir Eksperimen 4v2", "",
        "## Desain dan ruang lingkup",
        "Dataset E4v2 memuat 1.668 citra dari 1.284 pasien, dengan 1.411 citra "
        "development (478 DS-TB; 933 DR-TB) dan 257 citra testing "
        "(81 DS-TB; 176 DR-TB). Data development dibagi menggunakan "
        "StratifiedGroupKFold berbasis pasien. Empat konfigurasi M1–M4 "
        "menghasilkan 20 training. Prediksi testing merupakan rerata probabilitas "
        "dari lima checkpoint terbaik untuk masing-masing konfigurasi.",
        "", "## Validasi OOF", "",
        "| Model | OOF AUROC | Mean ± SD AUROC Fold | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        q=oof.loc[m]
        lines.append(f"| {m} — {NAMES[m]} | {q.oof_auroc:.4f} | "
            f"{q.fold_auroc_mean:.4f} ± {q.fold_auroc_sd_sample:.4f} | "
            f"{q.oof_macro_f1:.4f} | {q.oof_balanced_accuracy:.4f} | "
            f"{q.oof_sensitivity_dr:.2%} | {q.oof_specificity_ds:.2%} |")
    lines += ["", "M4 dipilih berdasarkan pooled OOF AUROC sebelum testing. "
              "SD fold adalah simpangan baku sampel antarfold, bukan confidence interval.",
              "", "## Ensemble Testing", "",
              "| Model | Test AUROC | Macro-F1 | Balanced Accuracy | Sensitivity | Specificity | TN | FP | FN | TP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",]
    for m in MODELS:
        q=test.loc[m]
        lines.append(f"| {m} — {NAMES[m]} | {q.auroc:.4f} | {q.macro_f1:.4f} | "
                     f"{q.balanced_accuracy:.4f} | {q.sensitivity_dr:.2%} | "
                     f"{q.specificity_ds:.2%} | {int(q.tn)} | {int(q.fp)} | "
                     f"{int(q.fn)} | {int(q.tp)} |")
    m4=test.loc["M4"]
    lines += ["", "## Analisis model utama", "",
              f"M4 menghasilkan test AUROC {m4.auroc:.4f}. Pada threshold 0,5, "
              f"sensitivity DR-TB {m4.sensitivity_dr:.2%} dan specificity DS-TB "
              f"{m4.specificity_ds:.2%}; ditemukan {int(m4.fn)} false negative "
              f"dan {int(m4.fp)} false positive.",
              "", "## Training dan overfitting", "",
              "Perubahan dihitung per run dari epoch checkpoint AUROC terbaik hingga epoch terakhir.",
              "", "| Model | Mean accuracy gap @best | Mean Δ validation loss | Mean Δ validation AUROC |",
              "|---|---:|---:|---:|",]
    avg=records.groupby("model_code").agg(
        gap=("accuracy_gap_best","mean"),
        dloss=("val_loss_change_best_to_last","mean"),
        dauc=("val_auroc_change_best_to_last","mean"))
    for m in MODELS:
        q=avg.loc[m]
        lines.append(f"| {m} — {NAMES[m]} | {q.gap:+.4f} | {q.dloss:+.4f} | {q.dauc:+.4f} |")
    lines += ["", "Pola training loss yang terus turun disertai peningkatan validation loss "
              "dapat mengindikasikan overfitting; hal tersebut tidak serta-merta "
              "menunjukkan bahwa checkpoint terbaik gagal melakukan generalisasi.",
              "", "## Batasan interpretasi", "",
              "- OOF berasal dari prediksi model fold individual, sedangkan testing "
              "menggunakan ensemble lima model pada kelompok citra berbeda.",
              "- Checkpoint terbaik dipilih menggunakan fold validation yang sama dengan "
              "perhitungan OOF; estimasi OOF dapat mengandung bias seleksi.",
              "- Test set E4v2 merupakan test set E4 asli yang pernah digunakan; "
              "evaluasi ini bersifat eksploratif, bukan holdout prospektif baru.",
              "- Dataset development tidak seimbang: 933 DR-TB berbanding 478 DS-TB. "
              "Kondisi ini penting saat menafsirkan sensitivity dan specificity.",
              "- Semua metrik dihitung pada tingkat citra, bukan tingkat pasien.",
              "- Grafik tidak mengubah threshold, model utama, checkpoint, atau bobot model.",
              "", "## Daftar Gambar", ""]
    lines.extend(f"- `{x}`" for x in figs)
    return "\n".join(lines)+"\n"


def run_checks():
    oof,test,oof_lock,test_lock,source_hashes=source_data()
    histories,records,history_hashes=histories_data(oof,oof_lock)
    return oof,test,oof_lock,test_lock,source_hashes,histories,records,history_hashes


def run():
    oof,test,oof_lock,test_lock,sources,hist,records,hhash=run_checks()
    require(not OUT.exists(),
            f"Folder output sudah ada, tidak akan ditimpa: {OUT}\nGunakan --verify untuk pemeriksaan.")
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        figs=charts(oof,test,hist,records,OUT)
        records.to_csv(OUT/"08_overfitting_per_fold.csv",index=False,float_format="%.12g")
        rows=[]
        for m in MODELS:
            q=oof.loc[m];r=test.loc[m]
            rows.append({
                "model_code":m,"model":STEMS[m],"primary_preselected":m=="M4",
                "oof_auroc":q.oof_auroc,
                "fold_auroc_mean":q.fold_auroc_mean,
                "fold_auroc_sd_sample":q.fold_auroc_sd_sample,
                "oof_macro_f1":q.oof_macro_f1,
                "oof_balanced_accuracy":q.oof_balanced_accuracy,
                "oof_sensitivity_dr":q.oof_sensitivity_dr,
                "oof_specificity_ds":q.oof_specificity_ds,
                "test_auroc":r.auroc,"test_macro_f1":r.macro_f1,
                "test_balanced_accuracy":r.balanced_accuracy,
                "test_sensitivity_dr":r.sensitivity_dr,
                "test_specificity_ds":r.specificity_ds,
                "test_tn":int(r.tn),"test_fp":int(r.fp),
                "test_fn":int(r.fn),"test_tp":int(r.tp),
            })
        pd.DataFrame(rows).to_csv(OUT/"08_e4v2_final_comparison.csv",index=False,float_format="%.12g")
        (OUT/"08_analisis_akhir_e4v2.md").write_text(
            analysis_markdown(oof,test,records,figs),encoding="utf-8")
        files=sorted(p for p in OUT.iterdir() if p.is_file())
        audit={
            "status":"PASS_E4V2_ACADEMIC_FINAL_VISUALIZATION",
            "experiment":"E4v2","model_primary":"M4","n_charts":len(figs),
            "n_training_histories_verified":len(hist),
            "development_images":1411,"test_images":257,
            "test_used_previously_in_e4":True,
            "no_training_or_inference":True,
            "no_images_or_checkpoints_loaded":True,
            "source_sha256":sources,
            "history_sha256":hhash,
            "outputs_sha256":{p.name:sha256(p) for p in files},
        }
        (OUT/"08_visualization_audit_report.json").write_text(
            json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    except Exception:
        print("Output parsial disimpan untuk diagnosis:",OUT,file=sys.stderr)
        raise
    print(f"PASS_E4V2_ACADEMIC_FINAL_VISUALIZATION | figures={len(figs)} | histories={len(hist)}")
    print("Output:",OUT)


def verify():
    oof,test,oof_lock,test_lock,sources,hist,records,hhash=run_checks()
    audit=read_json(OUT/"08_visualization_audit_report.json")
    require(audit["status"]=="PASS_E4V2_ACADEMIC_FINAL_VISUALIZATION",
            "Status laporan visualisasi tidak sesuai")
    require(audit["source_sha256"]==sources,"Sumber berubah setelah visualisasi")
    require(audit["history_sha256"]==hhash,"History berubah setelah visualisasi")
    require(audit["n_charts"]==20 and audit["n_training_histories_verified"]==20,
            "Jumlah riwayat atau grafik tidak sesuai")
    for filename,digest in audit["outputs_sha256"].items():
        verify_hash(OUT/filename,digest)
    require(sorted(p.name for p in OUT.glob("*.png"))==sorted(EXPECTED_PNG),
            "Daftar grafik tidak lengkap")
    require(len(list(OUT.iterdir()))==len(audit["outputs_sha256"])+1,
            "Ada file tambahan/missing pada output terkunci")
    print("PASS_E4V2_ACADEMIC_FINAL_VISUALIZATION_VERIFIED | figures=20 | histories=20")
    print("Output:",OUT)


def main():
    ap=argparse.ArgumentParser(description="20 grafik akademis Eksperimen 4v2 dari hasil terkunci")
    action=ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--check",action="store_true",help="Audit sumber sebelum pembuatan grafik")
    action.add_argument("--run",action="store_true",help="Hasilkan 20 grafik dan laporan")
    action.add_argument("--verify",action="store_true",help="Verifikasi seluruh output")
    args=ap.parse_args()
    if args.check:
        oof,test,oof_lock,test_lock,sources,hist,records,hhash=run_checks()
        print("PASS_E4V2_ACADEMIC_SOURCE_AUDIT | 20 histories | OOF + testing verified")
        print("Primary M4 | OOF AUROC=%.6f | test AUROC=%.6f"%
              (oof.loc["M4","oof_auroc"],test.loc["M4","auroc"]))
        if OUT.exists():
            print("CATATAN: Folder output sudah ada. Gunakan --verify; --run tidak akan menimpanya.")
        print("Tidak ada training, inferensi, atau akses ke citra.")
    elif args.run: run()
    elif args.verify: verify()


if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL_E4V2_VISUALIZATION: {type(exc).__name__}: {exc}",file=sys.stderr)
        raise SystemExit(1)
