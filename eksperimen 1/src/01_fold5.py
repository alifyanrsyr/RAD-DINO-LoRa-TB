from __future__ import annotations
import argparse, hashlib, json, os, sys, shutil, tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import sklearn

SOURCE_FILES = {
 'split': 'data/manifests/experiment4_split_70_15_15_seed42.csv',
 'old_cv': 'data/manifests/experiment4_train_5fold_seed42.csv',
 'processed': 'data/processed/experiment4_letterbox518_v1/processed_manifest.csv',
 'audit': 'data/audit/processed_dataset_final_audit/processed_png_audit_report.json'
}
KEY = ['source_archive', 'archive_member_path']
META = ['patient_id', 'y_true', 'source_zip', 'split', 'cv_fold']
N_CLASSES = {0: 'DS-TB', 1: 'DR-TB'}
EXPECTED = {
 'train': (1169, 898, 406, 763),
 'validation': (242, 193, 72, 170),
 'test': (257, 193, 81, 176),
}
FOLDER = 'experiment4_fixedtest257_dev1411_5fold_seed42_v1'


def require(ok, text):
    if not bool(ok):
        raise RuntimeError(text)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def csvdump(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8')


def class_counts(frame):
    return dict(images=int(len(frame)), patients=int(frame.patient_id.nunique()),
                ds_images=int(frame.y_true.eq('0').sum()),
                dr_images=int(frame.y_true.eq('1').sum()),
                ds_patients=int(frame.loc[frame.y_true.eq('0'),'patient_id'].nunique()),
                dr_patients=int(frame.loc[frame.y_true.eq('1'),'patient_id'].nunique()))


def unique_and_complete(df, where):
    require(not df[KEY].duplicated().any(), f'{where}: duplicate DICOM identity')
    for col in KEY + ['patient_id','y_true']:
        require(df[col].ne('').all(), f'{where}: empty {col}')
    require(df.y_true.isin(['0','1']).all(), f'{where}: labels outside 0/1')
    require(df.groupby('patient_id').y_true.nunique().max() == 1,
            f'{where}: one patient has conflicting labels')


def read_source(e4):
    p = {k:e4/v for k,v in SOURCE_FILES.items()}
    for k,path in p.items():
        require(path.is_file(), f'Missing E4 {k}: {path}')
    audit = json.loads(p['audit'].read_text(encoding='utf-8'))
    require(audit.get('status') == 'PASS' and audit.get('n_errors') == 0 and
            audit.get('n_checked_images') == 1668,
            'Original E4 PNG audit is not PASS for 1668 images')
    expected_hashes = {
        'split': audit['locked_manifest_sha256']['split'],
        'old_cv': audit['locked_manifest_sha256']['fold'],
        'processed': audit['processed_manifest_sha256']
    }
    hashes = {k:sha(path) for k,path in p.items()}
    for k,expect in expected_hashes.items():
        require(hashes[k] == expect,
                f'Original E4 {k} SHA256 mismatch! Expected {expect}; found {hashes[k]}')
    split = pd.read_csv(p['split'], dtype=str, keep_default_na=False)
    proc = pd.read_csv(p['processed'], dtype=str, keep_default_na=False)
    old_cv = pd.read_csv(p['old_cv'], dtype=str, keep_default_na=False)
    for name,df in [('split',split),('processed',proc),('old_cv',old_cv)]:
        must = KEY + META if name!='processed' else KEY+META+['raw_path','clahe_path']
        require(set(must).issubset(df.columns), f'{name}: missing columns {set(must)-set(df.columns)}')
        unique_and_complete(df,name)
    require(len(split)==1668 and len(proc)==1668 and len(old_cv)==1169,
            'E4 dataset source sizes differ from locked protocol')
    require(set(split.split)=={'train','validation','test'}, 'E4 original split role mismatch')
    merged = proc.merge(split[KEY+META], on=KEY, how='outer', indicator=True,
                        validate='one_to_one', suffixes=('_p','_s'))
    require(len(merged)==1668 and merged['_merge'].eq('both').all(),
            'E4 processed and split DICOM identities do not match')
    for field in META:
        require(merged[field+'_p'].eq(merged[field+'_s']).all(),
                f'Original E4 {field} metadata mismatch')
    for subset,(ni,np_,nds,ndr) in EXPECTED.items():
        s=proc.loc[proc.split.eq(subset)]
        counts=class_counts(s)
        require((counts['images'],counts['patients'],counts['ds_images'],counts['dr_images'])
                == (ni,np_,nds,ndr), f'E4 {subset} count mismatch: {counts}')
    require(proc.patient_id.nunique()==1284, 'Original E4 patient count mismatch')
    patient_roles=proc.groupby('patient_id').split.nunique()
    require(patient_roles.max()==1, 'E4 patient leakage across original splits')
    old=proc[proc.split.eq('train')][KEY+META].merge(old_cv[KEY+META], on=KEY, how='outer',
                           validate='one_to_one',indicator=True,suffixes=('_p','_c'))
    require(len(old)==1169 and old['_merge'].eq('both').all(), 'E4 old fold mapping mismatch')
    for field in META:
        require(old[field+'_p'].eq(old[field+'_c']).all(),
                f'E4 old CV {field} mismatch')
    return proc, hashes


def build_folds(source):
    dev=source.loc[source.split.isin(['train','validation'])].copy()
    test=source.loc[source.split.eq('test')].copy()
    require(len(dev)==1411 and dev.patient_id.nunique()==1091, 'E4 85% development size mismatch')
    require(len(test)==257 and test.patient_id.nunique()==193, 'E4 fixed test size mismatch')
    require(not (set(dev.patient_id)&set(test.patient_id)), 'Development/test patient leakage')
    require(dev.y_true.eq('0').sum()==478 and dev.y_true.eq('1').sum()==933,
            'E4 development class counts mismatch')
    require(test.y_true.eq('0').sum()==81 and test.y_true.eq('1').sum()==176,
            'E4 testing class counts mismatch')
    dev=dev.sort_values(KEY,kind='mergesort').reset_index(drop=True)
    dev=dev.rename(columns={'split':'e4_original_split','cv_fold':'e4_original_cv_fold'})
    dev['split']='development'
    dev['cv_fold']=''
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    labels=dev.y_true.astype(int).to_numpy()
    group=dev.patient_id.to_numpy()
    for i,(_,va) in enumerate(splitter.split(np.zeros(len(dev)),labels,group),1):
        require(dev.loc[va,'cv_fold'].eq('').all(), 'An image assigned to more than one CV fold')
        dev.loc[va,'cv_fold']=str(i)
    require(dev.cv_fold.isin(['1','2','3','4','5']).all(), 'CV assignment incomplete')
    require(dev.groupby('patient_id').cv_fold.nunique().max()==1,
            'Patient images cross CV validation folds')
    fold_count=[]
    for i in range(1,6):
        v=dev.loc[dev.cv_fold.eq(str(i))]
        t=dev.loc[dev.cv_fold.ne(str(i))]
        require(not(set(t.patient_id)&set(v.patient_id)), f'Patient leakage in fold {i}')
        for subset,fr in [('train',t),('validation',v)]:
            cnt=class_counts(fr)
            require(cnt['ds_images']>0 and cnt['dr_images']>0,
                    f'Fold {i} {subset} missing a class')
            fold_count.append({'fold':i,'subset':subset,**cnt})
    patient=dev[['patient_id','y_true','cv_fold']].drop_duplicates().sort_values('patient_id')
    require(len(patient)==1091, 'Patient assignment table count mismatch')
    return dev, test, patient, pd.DataFrame(fold_count)


def write_output(stage, dev, test, patient, counts, source_hashes):
    stage.mkdir(parents=True,exist_ok=False)
    csvdump(stage/'development_1411_5fold_LOCKED.csv',dev)
    csvdump(stage/'patient_fold_assignments.csv',patient)
    csvdump(stage/'fold_class_patient_counts.csv',counts)
    for i in range(1,6):
        fold_dir=stage/f'fold_{i}'
        fold_dir.mkdir()
        for role,is_train in [('train',True),('validation',False)]:
            mask=dev.cv_fold.ne(str(i)) if is_train else dev.cv_fold.eq(str(i))
            frame=dev.loc[mask].copy()
            frame['split']=role
            csvdump(fold_dir/f'{role}.csv',frame)
    reference={
      'policy':'REFERENCE ONLY; do not access test pixels, predictions or labels for model selection',
      'source_e4_split_sha256':source_hashes['split'],
      'test_images':int(len(test)), 'test_patients':int(test.patient_id.nunique()),
      'test_ds_images':int(test.y_true.eq('0').sum()),
      'test_dr_images':int(test.y_true.eq('1').sum()),
      'test_original_split':'test',
      'test_previously_evaluated_in_original_E4':True,
      'test_pixel_accessed':False,
    }
    jdump(stage/'fixed_test_REFERENCE_NO_PIXELS.json',reference)
    outputs={}
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            outputs[path.relative_to(stage).as_posix()]=sha(path)
    report={
      'status':'PASS_E4_CV85_FOLDS_LOCKED',
      'experiment':'experiment4_cv85_protocol_aligned_to_e6',
      'source_e4_images':1668,'source_e4_patients':1284,
      'development':class_counts(dev),
      'test':class_counts(test),
      'actual_development_percentage':100*len(dev)/1668,
      'actual_holdout_percentage':100*len(test)/1668,
      'old_e4_training_pool_images':1169,
      'old_e4_fixed_validation_images_now_in_cv':242,
      'algorithm':'StratifiedGroupKFold',
      'n_splits':5,'seed':42,'sklearn_version':sklearn.__version__,
      'class_ids':N_CLASSES,
      'same_test_patients_as_original_e4':True,
      'no_patient_leakage':True,
      'each_development_image_oof_once':True,
      'each_development_patient_oof_once':True,
      'train_or_inference_performed':False,
      'test_pixel_accessed':False,
      'source_sha256':source_hashes,
      'output_sha256':outputs,
      'fold_counts':counts.to_dict(orient='records'),
      'caution':'Test was previously exposed in E4. Exploratory reuse; E4 and E6 tests are different patients/cohorts.',
      'next_stage':'Train M1-M4 on these five folds using exact E6 model hyperparameters, not E7 hyperparameters.'
    }
    jdump(stage/'01_e4_cv85_fold_lock_report.json',report)
    return report


def verify(out,e4):
    require(out.exists(),f'Output not found: {out}')
    report_path=out/'01_e4_cv85_fold_lock_report.json'
    require(report_path.is_file(),'Missing lock report')
    report=json.loads(report_path.read_text(encoding='utf-8'))
    require(report['status']=='PASS_E4_CV85_FOLDS_LOCKED', 'Unexpected lock report status')
    proc, hashes = read_source(e4)
    require(hashes==report['source_sha256'], 'Original E4 source hashes changed')
    dev, test, patients, counts=build_folds(proc)
    refcsv=out/'development_1411_5fold_LOCKED.csv'
    actual=pd.read_csv(refcsv,dtype=str,keep_default_na=False)
    require(actual.equals(dev), 'Locked dev rows differ from deterministic recomputation')
    require(pd.read_csv(out/'patient_fold_assignments.csv',dtype=str,keep_default_na=False)
            .equals(patients.reset_index(drop=True)), 'Patient fold assignment mismatch')
    for k,h in report['output_sha256'].items():
        path=out/k
        require(path.is_file() and sha(path)==h,f'Missing or modified output: {k}')
    for i in range(1,6):
        for role in ['train','validation']:
            subset=pd.read_csv(out/f'fold_{i}'/f'{role}.csv',dtype=str,keep_default_na=False)
            expected=dev.loc[dev.cv_fold.ne(str(i)) if role=='train' else
                             dev.cv_fold.eq(str(i))].copy()
            expected['split']=role
            expected=expected.reset_index(drop=True)
            require(subset.equals(expected),f'Fold {i} {role}: unexpected manifest contents')
    print('PASS_E4_CV85_FOLD_LOCK_VERIFIED | 1411 development | 257 test (reference only) | 5 folds')
    print(f'  Output: {out}')


def main():
    ap=argparse.ArgumentParser(description='E4 original holdout fixed, CV E6 style on train+val')
    modes=ap.add_mutually_exclusive_group(required=True)
    for mode in ['check','run','verify']:
        modes.add_argument('--'+mode,action='store_true')
    script_dir = Path(__file__).resolve().parent
    # Support a script located directly in eksperimen4v2/ or in eksperimen4v2/src/.
    output_experiment_dir = script_dir.parent if script_dir.name.lower() == 'src' else script_dir
    project = output_experiment_dir.parent
    ap.add_argument('--project-root', type=Path, default=project)
    args = ap.parse_args()
    project = args.project_root.resolve()
    require(output_experiment_dir.name.lower() == 'eksperimen4v2',
            'Run this adjusted script in eksperimen4v2/ (or eksperimen4v2/src/)')
    require(project.name == 'RAD-DINO-TB' or (project/'eksperimen4').is_dir(),
            f'Project root must contain eksperimen4/: {project}')
    e4 = project/'eksperimen4'
    out = project/'eksperimen4v2'/'data'/'manifests'/FOLDER
    if args.verify:
        verify(out,e4)
        return
    proc, hashes=read_source(e4)
    dev, test, patients, counts=build_folds(proc)
    print('PASS_E4_SOURCE_LOCK | 1668 images / 1284 patients | E4 pixel audit PASS')
    print('E4 development: 1411 images / 1091 patients | DS=478 DR=933')
    print('E4 preserved test: 257 images / 193 patients | DS=81 DR=176')
    print('Actual dev/test proportions: 84.59% / 15.41%')
    print(counts[['fold','subset','images','patients','ds_images','dr_images']].to_string(index=False))
    if args.check:
        print('DRY CHECK PASS_E4_CV85 | No new files or model inference')
        return
    require(not out.exists(),f'Locked output already exists: {out}. Use --verify, never overwrite.')
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='._e4_cv85_staging_',dir=out.parent))
    try:
        shutil.rmtree(temp)
        report=write_output(temp,dev,test,patients,counts,hashes)
        temp.rename(out)
    except Exception:
        print(f'ERROR: inspect staging folder: {temp}',file=sys.stderr)
        raise
    print(report['status'],'Output:',out)
    verify(out,e4)

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f'FAIL_E4_CV85: {type(e).__name__}: {e}',file=sys.stderr)
        raise SystemExit(1)
