from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
# SHA-256 values intentionally pinned to audited Experiment 5 results.
STAGE3_SHA = 'da4aff5e6a628152925a6e2472ba86b470b9dba5f95fd4e1b7f730dcb055c26d'
STAGE4_SHA = 'f67ee2747006aaecb03751b64fd6b0a944e157d5a81f762212b2926ee94fce0b'
STAGE5_SHA = '7f17b9622f066347c7951314e12ad73a9b323929587b420e97538c2f38d7a457'
STAGE5_REPORT_SHA = '95c25b845f8b222eb65d77f9a626754b172968b0d4ce5cfb93b84a21b5dcf750'
STAGE5_PNG_AUDIT_SHA = 'c334f938382d6a65c8469c2fcc81e4b6e882c8fc0a74da9a50bbc25e68f4ab6f'
KEY = ['source_archive', 'archive_member_path']
META = KEY + ['source_zip','patient_id','y_true','split']
CLASS_COUNTS = {'train': {0:777,1:777},'validation':{0:167,1:166},'test':{0:166,1:167}}
IMAGE_COUNTS = {'train':1554,'validation':333,'test':333}
PATIENT_COUNTS = {'train':1211,'validation':260,'test':259}
DEFAULT_OUTPUT_NAME = 'experiment6_85dev_15test_5fold_seed42_v1'


def require(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)


def sha256(fp):
    h=hashlib.sha256()
    with open(fp,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def jread(path):
    require(path.is_file(),f'File wajib tidak ditemukan: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def cread(path):
    require(path.is_file(),f'CSV wajib tidak ditemukan: {path}')
    return pd.read_csv(path,dtype=str,keep_default_na=False,low_memory=False)


def same_rows(a,b,cols):
    return a[cols].sort_values(KEY).reset_index(drop=True).equals(
        b[cols].sort_values(KEY).reset_index(drop=True))


def guarded_png_path(png_dir,relative,variant):
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts and len(p.parts)==2,
            f'Path PNG tidak aman: {relative}')
    require(p.parts[0]==f'{variant}_518' and p.suffix.lower()=='.png',
            f'Path {variant} tidak sesuai: {relative}')
    return png_dir/p


def inspect_e5(exp5, check_png=True):
    """Verify actual experiment5 artifacts and IDs; NEVER read testing pixels."""
    a=exp5/'audits'/'03_final_lock'
    b=exp5/'data'/'manifests'/'experiment5_split_70_15_15_seed42_v1'
    c=exp5/'data'/'processed'/'experiment5_letterbox518_v1'
    files={
        'stage3': a/'experiment5_dataset_2220_LOCKED_UNSPLIT.csv',
        'stage3_report': a/'03_final_lock_report.json',
        'stage4': b/'experiment5_split_70_15_15_seed42_LOCKED.csv',
        'stage4_report': b/'04_split_lock_report.json',
        'train': b/'train.csv', 'validation': b/'validation.csv', 'test':b/'test.csv',
        'stage5': c/'processed_manifest.csv',
        'stage5_report': c/'preprocessing_report.json',
        'stage5_audit': c/'05_png_technical_audit_report.json',
    }
    for name,path in files.items():
        require(path.is_file(),f'Input Eksperimen 5 tidak ditemukan [{name}]: {path}')
    r3=jread(files['stage3_report'])
    r4=jread(files['stage4_report'])
    r5=jread(files['stage5_report'])
    ra=jread(files['stage5_audit'])
    require(r3.get('status')=='PASS_FINAL_LOCKED','Stage3 Eksperimen 5 tidak PASS')
    require(r4.get('status')=='PASS_PATIENT_LEVEL_SPLIT_LOCKED','Stage4 Eksperimen 5 tidak PASS')
    require(r4.get('seed')==SEED and r4.get('no_patient_overlap') is True,
            'Seed/patient split Eksperimen 5 tidak sesuai')
    require(r5.get('status')=='PASS_PREPROCESSING' and r5.get('n_success')==2220,
            'Stage5 preprocessing belum PASS 2220')
    require(ra.get('status')=='PASS_PNG_TECHNICAL_AUDIT' and ra.get('png_verified')==4440,
            'Audit PNG Eksperimen 5 belum PASS 4440')
    require(sha256(files['stage3'])==STAGE3_SHA==r3.get('locked_manifest_sha256'),
            'Hash locked dataset E5 tidak cocok')
    require(sha256(files['stage4'])==STAGE4_SHA==r4.get('output_sha256',{}).get(files['stage4'].name),
            'Hash split LOCKED E5 tidak cocok')
    require(sha256(files['stage5'])==STAGE5_SHA==r5.get('processed_manifest_sha256'),
            'Hash processed_manifest E5 tidak cocok')
    require(sha256(files['stage5_report'])==STAGE5_REPORT_SHA,
            'Hash preprocessing report E5 tidak cocok')
    require(sha256(files['stage5_audit'])==STAGE5_PNG_AUDIT_SHA,
            'Hash audit PNG E5 tidak cocok')
    require(sha256(files['stage3_report'])==r4.get('source_final_lock_sha256'),
            'Rantai checksum E5 Stage3/Stage4 terputus')
    for name in ('train','validation','test'):
        fp=files[name]
        require(sha256(fp)==r4.get('output_sha256',{}).get(fp.name),
                f'Hash berkas E5 {fp.name} tidak cocok')
    orig=cread(files['stage3'])
    split=cread(files['stage4'])
    proc=cread(files['stage5'])
    require(len(orig)==len(split)==len(proc)==2220,'Manifest tidak sama-sama 2220 citra')
    require(set(META).issubset(split.columns) and set(META+['raw_path','clahe_path']).issubset(proc.columns),
            'Kolom penting manifest E5 tidak lengkap')
    require('cv_fold' not in split and 'cv_fold' not in proc,
            'Sumber Eksperimen 5 tiba-tiba sudah memiliki cv_fold')
    require(not split[KEY].duplicated().any() and not proc[KEY].duplicated().any(),
            'Duplikasi DICOM pada manifest E5')
    require(same_rows(orig,split,[c for c in orig.columns if c in split.columns]),
            'Identitas sumber dataset E5 berubah setelah split')
    require(same_rows(split,proc,META),'Processed PNG tidak identik dengan split E5')
    require(split.y_true.isin(('0','1')).all(),'Label bukan 0/1')
    require(split.split.isin(IMAGE_COUNTS).all(),'Nama split E5 tidak dikenal')
    require(split.groupby('patient_id').y_true.nunique().eq(1).all(),
            'Konflik label per pasien E5')
    require(split.groupby('patient_id').split.nunique().eq(1).all(),
            'Patient leakage pada split E5')
    require(split.patient_id.nunique()==1730,'Pasien E5 bukan 1730')
    for name in IMAGE_COUNTS:
        sub=split.loc[split.split==name]
        source=cread(files[name])
        require(same_rows(sub,source,list(split.columns)),f'{name}.csv berbeda dari manifest locked')
        require(len(sub)==IMAGE_COUNTS[name] and sub.patient_id.nunique()==PATIENT_COUNTS[name],
                f'Jumlah citra/pasien pada {name} salah')
        for y,count in CLASS_COUNTS[name].items():
            require(int((sub.y_true==str(y)).sum())==count,
                    f'Jumlah kelas {y} pada {name} salah')
    require(sha256(files['stage4'])==r5.get('input_sha256',{}).get('stage4_split_manifest_sha256'),
            'Proses PNG terhubung ke split E5 yang berbeda')
    # Merge preserves order of locked split; never touches pixels or uses test labels for fold assignment.
    merged=split.merge(proc[KEY+['raw_path','clahe_path']],on=KEY,how='left',validate='one_to_one',sort=False)
    require(len(merged)==2220 and merged.raw_path.ne('').all() and merged.clahe_path.ne('').all(),
            'Tidak semua citra punya path RAW/CLAHE')
    dev=merged.loc[merged.split.isin(('train','validation'))].copy().reset_index(drop=True)
    test=merged.loc[merged.split.eq('test')].copy().reset_index(drop=True)
    require(len(dev)==1887 and dev.patient_id.nunique()==1471,'Development bukan 1887/1471')
    require(len(test)==333 and test.patient_id.nunique()==259,'Testing bukan 333/259')
    require(not set(dev.patient_id).intersection(set(test.patient_id)),
            'KEBOCORAN pasien development-test')
    require(int(dev.y_true.eq('0').sum())==944 and int(dev.y_true.eq('1').sum())==943,
            'Kelas development salah (harus DS944 DR943)')
    require(dev.groupby('patient_id').y_true.nunique().eq(1).all(),'Konflik kelas development')
    if check_png:
        for variant in ('raw','clahe'):
            for relative in dev[f'{variant}_path']:
                path=guarded_png_path(c,relative,variant)
                require(path.is_file(),f'PNG DEVELOPMENT tidak ditemukan: {path}')
    fingerprints={name:sha256(path) for name,path in files.items()}
    return dev,test,files,fingerprints,c


def create_assignments(dev):
    # StratifiedGroupKFold operates on development ONLY; no test input to .split().
    cv=StratifiedGroupKFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)
    vfold=np.zeros(len(dev),dtype=np.int8)
    for fold,(itrain,ival) in enumerate(cv.split(X=np.zeros(len(dev),dtype=np.int8),
                                                 y=dev['y_true'].to_numpy(),
                                                 groups=dev['patient_id'].to_numpy()),start=1):
        require(not set(dev.iloc[itrain].patient_id).intersection(dev.iloc[ival].patient_id),
                f'Patient overlap dalam fold {fold}')
        require(np.all(vfold[ival]==0),f'Citra menjadi VAL lebih dari sekali di fold {fold}')
        vfold[ival]=fold
    require(np.all(vfold>0),'Tidak semua citra development masuk validation tepat satu kali')
    out=dev.copy()
    out=out.rename(columns={'split':'e5_original_split'})
    out['cv_fold']=vfold.astype(str)
    require(out.groupby('patient_id').cv_fold.nunique().eq(1).all(),
            'Pasien dengan citra tersebar di beberapa validation fold')
    return out


def audit_folds(devfold,test):
    require(len(devfold)==1887 and len(test)==333,'Total development/holdout berubah')
    require(devfold.cv_fold.isin(tuple(str(x) for x in range(1,6))).all(),
            'Fold hanya boleh 1..5')
    require(devfold.groupby('patient_id').cv_fold.nunique().eq(1).all(),
            'Pasien berpindah-pindah fold')
    require(not set(devfold.patient_id).intersection(test.patient_id),
            'Pasien test muncul pada CV')
    require(devfold[KEY].duplicated().sum()==0,'Duplicate image in CV')
    require(devfold.groupby('cv_fold').patient_id.nunique().sum()==1471,
            'Pasien OOF tidak muncul tepat sekali')
    rows=[]
    for fold in range(1,N_FOLDS+1):
        is_val=devfold.cv_fold.eq(str(fold))
        tr=devfold.loc[~is_val]
        va=devfold.loc[is_val]
        require(len(tr)+len(va)==1887,'Fold counts do not sum to development')
        require(not set(tr.patient_id).intersection(va.patient_id),
                f'Patient leakage F{fold}')
        require(tr.y_true.nunique()==va.y_true.nunique()==2,f'Fold {fold} lacks one class')
        for part,sub in (('train',tr),('validation',va)):
            rows.append({'fold':fold,'subset':part,'images':len(sub),
                         'patients':sub.patient_id.nunique(),
                         'ds_images':int(sub.y_true.eq('0').sum()),
                         'dr_images':int(sub.y_true.eq('1').sum()),
                         'ds_patients':sub.loc[sub.y_true.eq('0'),'patient_id'].nunique(),
                         'dr_patients':sub.loc[sub.y_true.eq('1'),'patient_id'].nunique()})
    return pd.DataFrame(rows)


def result_manifest(devfold,fold,part):
    is_val=devfold.cv_fold.eq(str(fold))
    d=devfold.loc[is_val if part=='validation' else ~is_val].copy()
    d['split']=part
    return d


def write_outputs(dst,devfold,counts,source_files,hashes,png_root,exp5):
    dst.mkdir(parents=True,exist_ok=False)
    devfold.to_csv(dst/'e6_development_1887_5fold_LOCKED.csv',index=False)
    pp=devfold.groupby('patient_id',as_index=False).agg(
        y_true=('y_true','first'),n_images=('y_true','size'),cv_fold=('cv_fold','first'),
        e5_original_split=('e5_original_split','first'))
    pp=pp.sort_values(['cv_fold','patient_id'])
    pp.to_csv(dst/'e6_patient_fold_assignments.csv',index=False)
    counts.to_csv(dst/'e6_fold_class_patient_counts.csv',index=False)
    for f in range(1,N_FOLDS+1):
        folder=dst/f'fold_{f}'
        folder.mkdir(exist_ok=False)
        for p in ('train','validation'):
            result_manifest(devfold,f,p).to_csv(folder/f'{p}.csv',index=False)
    holdout={
        'experiment':'experiment6',
        'policy':'REUSE E5 fixed 15% test UNCHANGED; NEVER include in 5-fold CV',
        'e5_test_manifest_path':str(source_files['test']),
        'e5_test_manifest_sha256':hashes['test'],
        'holdout_images':333,'holdout_patients':259,
        'model_training_must_not_load_holdout':True,
        'note':'This test set was already evaluated during E5; E6 comparison is exploratory, not a new unseen confirmatory holdout.'
    }
    (dst/'e6_holdout_TEST_REFERENCE_DO_NOT_TRAIN.json').write_text(
        json.dumps(holdout,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    filenames=[f for f in dst.rglob('*.csv')]
    filenames += [dst/'e6_holdout_TEST_REFERENCE_DO_NOT_TRAIN.json']
    file_hashes={str(f.relative_to(dst)).replace('\\','/'):sha256(f)
                 for f in sorted(filenames)}
    report={
        'status':'PASS_E6_FOLDS_LOCKED',
        'experiment':'experiment6',
        'protocol':'fixed E5 holdout 15%; 5-fold StratifiedGroupKFold on E5 train+validation 85%',
        'algorithm':'sklearn.model_selection.StratifiedGroupKFold',
        'n_splits':N_FOLDS,'shuffle':True,'random_state':SEED,
        'development_images':1887,'development_patients':1471,
        'development_class_images':{'DS-TB':944,'DR-TB':943},
        'holdout_images':333,'holdout_patients':259,
        'test_manifest_sha256':hashes['test'],
        'no_patient_leakage':True,
        'oof_each_development_image_exactly_once':True,
        'oof_each_development_patient_exactly_once':True,
        'cv_input_contains_only_e5_train_and_validation':True,
        'test_pixels_read':False,'test_inference_performed':False,
        'training_or_finetuning_performed':False,
        'e5_source_folder':str(exp5),
        'e5_png_root':str(png_root),
        'source_sha256':hashes,
        'fold_counts':counts.to_dict(orient='records'),
        'output_sha256':file_hashes,
        'next_step':'Verify this report, then validate E6 DataLoader before training 4 configurations x 5 folds.',
    }
    (dst/'02_e6_5fold_lock_report.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def verify_output(dst):
    report=jread(dst/'02_e6_5fold_lock_report.json')
    require(report.get('status')=='PASS_E6_FOLDS_LOCKED',
            'Output lock report not PASS')
    for rel,expected in report['output_sha256'].items():
        require('..' not in Path(rel).parts and not Path(rel).is_absolute(),
                'Malicious output path')
        path=dst/Path(rel)
        require(path.is_file() and sha256(path)==expected,
                f'Output hash mismatch: {rel}')
    df=cread(dst/'e6_development_1887_5fold_LOCKED.csv')
    holdout=jread(dst/'e6_holdout_TEST_REFERENCE_DO_NOT_TRAIN.json')
    test_path=Path(holdout['e5_test_manifest_path'])
    require(test_path.is_file() and sha256(test_path)==holdout['e5_test_manifest_sha256'],
            'E5 fixed test file changed')
    test=cread(test_path)
    counts=audit_folds(df,test)
    existing=cread(dst/'e6_fold_class_patient_counts.csv')
    require(counts.astype(str).equals(existing.astype(str)),
            'Recorded fold counts differ')
    for fold in range(1,6):
        for part in ('train','validation'):
            d=cread(dst/f'fold_{fold}'/f'{part}.csv')
            expected=result_manifest(df,fold,part).reset_index(drop=True)
            require(d.equals(expected),'Fold file content differs from development assignment')
    return report


def main():
    p=argparse.ArgumentParser(description='Eksperimen 6 — 85% development 5-fold, 15% fixed holdout')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Read-only preflight; no outputs')
    mode.add_argument('--run',action='store_true',help='Create locked E6 fold manifests once')
    mode.add_argument('--verify',action='store_true',help='Verify generated outputs and hashes')
    p.add_argument('--e5-root',type=Path,default=None,help='Optional E5 folder; default sibling eksperimen5')
    p.add_argument('--out',type=Path,default=None,help='Optional new output folder; never overwrite')
    args=p.parse_args()
    e6=Path(__file__).resolve().parent.parent
    e5=(args.e5_root or e6.parent/'eksperimen5').resolve()
    output=(args.out or e6/'data'/'manifests'/DEFAULT_OUTPUT_NAME).resolve()
    require(output!=e5 and e5 not in output.parents,'Output must NOT be within Experiment 5')
    if args.verify:
        require(output.is_dir(),f'Output folder missing: {output}')
        rpt=verify_output(output)
        print('VERIFY PASS_E6_FOLDS_LOCKED | development=1887 test reference=333')
        print('Report:',output/'02_e6_5fold_lock_report.json')
        return 0
    require(e5.is_dir(),f'Folder eksperimen5 tidak ditemukan: {e5}')
    require(not output.exists(),f'Refusing to overwrite existing output: {output}')
    dev,test,files,hashes,pngroot=inspect_e5(e5,check_png=True)
    devfold=create_assignments(dev)
    counts=audit_folds(devfold,test)
    print('SOURCE E5 LOCKS PASS | development=1887 (1471 patients) | fixed test=333 (259 patients)')
    print('CV StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)')
    print(counts.to_string(index=False))
    if args.check:
        print('DRY CHECK PASS | no files modified; next command: --run')
        return 0
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='._e6_build_',dir=str(output.parent)))
    try:
        # write_outputs requires nonexistent dir; move to fresh child within tmp.
        staging=tmp/'content'
        write_outputs(staging,devfold,counts,files,hashes,pngroot,e5)
        verify_output(staging)
        require(not output.exists(),'Output appeared during creation; refusing overwrite')
        os.rename(staging,output)
        print('PASS_E6_FOLDS_LOCKED | output:',output)
        print('NEXT: python 01_02_buat_5fold_eksperimen6.py --verify')
        return 0
    finally:
        if tmp.exists():shutil.rmtree(tmp)


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'E6 STAGE 1-2 FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
