from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N = 5
SIZE = 518
MODEL = 'microsoft/rad-dino'
DIRNAME = 'experiment6_85dev_15test_5fold_seed42_v1'
KEY = ['source_archive','archive_member_path']
ID_COLS = KEY + ['patient_id','y_true','raw_path','clahe_path']
SOURCE_E5_SPLIT_SHA = 'f67ee2747006aaecb03751b64fd6b0a944e157d5a81f762212b2926ee94fce0b'
SOURCE_E5_PROCESSED_SHA = '7f17b9622f066347c7951314e12ad73a9b323929587b420e97538c2f38d7a457'
SOURCE_TEST_SHA = 'e62c588da8b406f6b8de90093f822c86620dd8701426d496de218f10ace7cc35'


def need(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def read_csv(path):
    need(path.is_file(), f'CSV tidak ditemukan: {path}')
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(path):
    need(path.is_file(), f'JSON tidak ditemukan: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def same_rows(left, right, columns):
    return left[columns].sort_values(KEY).reset_index(drop=True).equals(
        right[columns].sort_values(KEY).reset_index(drop=True))


def png_path(root, relative, variant):
    # E5 paths are always POSIX-style relative path e.g. raw_518/a0001.png.
    need(isinstance(relative, str), 'Nama file PNG bukan string')
    need('\\' not in relative and ':' not in relative, f'Path PNG mencurigakan: {relative}')
    rel = Path(relative)
    need(not rel.is_absolute() and len(rel.parts) == 2 and '..' not in rel.parts,
         f'Path PNG tidak aman: {relative}')
    need(rel.parts[0] == f'{variant}_518' and rel.suffix.lower() == '.png',
         f'Path varian {variant} salah: {relative}')
    return root / rel


def load_and_check(e6_root, e5_root):
    lock_dir = e6_root / 'data' / 'manifests' / DIRNAME
    report_file = lock_dir / '02_e6_5fold_lock_report.json'
    report = read_json(report_file)
    need(report.get('status') == 'PASS_E6_FOLDS_LOCKED', 'E6 fold belum dikunci/PASS')
    need(report.get('n_splits') == N and report.get('random_state') == 42,
         'n_splits/seed E6 berubah')
    need(report.get('development_images') == 1887 and report.get('development_patients') == 1471,
         'Development 1887 citra /1471 pasien tidak sesuai')
    need(report.get('holdout_images') == 333 and report.get('holdout_patients') == 259,
         'Holdout 333 citra/259 pasien tidak sesuai')
    need(report.get('no_patient_leakage') is True and
         report.get('oof_each_development_image_exactly_once') is True and
         report.get('oof_each_development_patient_exactly_once') is True,
         'Audit fold E6 tidak PASS')
    need(report.get('test_pixels_read') is False, 'Laporan fold menyatakan test pernah dibaca')
    need(report.get('source_sha256', {}).get('stage4') == SOURCE_E5_SPLIT_SHA and
         report.get('source_sha256', {}).get('stage5') == SOURCE_E5_PROCESSED_SHA and
         report.get('test_manifest_sha256') == SOURCE_TEST_SHA,
         'Sumber data E5 berbeda dari protokol terkunci')
    paths = {}
    for rel, expected in report.get('output_sha256', {}).items():
        safe = Path(rel)
        need(not safe.is_absolute() and '..' not in safe.parts and not rel.startswith(('\\','/')),
             f'Nama artefak tidak aman: {rel}')
        path = lock_dir / safe
        need(path.is_file() and checksum(path) == expected,
             f'File fold berubah/rusak: {path}')
        paths[rel] = path
    need(len(paths) == 14, f'Jumlah berkas output E6 bukan 14: {len(paths)}')
    dev = read_csv(lock_dir / 'e6_development_1887_5fold_LOCKED.csv')
    need(len(dev) == 1887 and dev.patient_id.nunique() == 1471,
         'Jumlah development berubah')
    need(set(ID_COLS + ['cv_fold','e5_original_split']).issubset(dev.columns),
         'Kolom wajib development hilang')
    need(not dev[KEY].duplicated().any(), 'Identitas DICOM development duplikat')
    need(dev.y_true.isin(('0','1')).all() and dev.cv_fold.isin(('1','2','3','4','5')).all(),
         'Label/fold di luar rentang')
    need(dev.e5_original_split.isin(('train','validation')).all(),
         'Ada test E5 masuk ke development')
    need(dev.groupby('patient_id').cv_fold.nunique().eq(1).all(),
         'Pasien berada pada lebih dari satu validation fold')
    need(dev.groupby('patient_id').y_true.nunique().eq(1).all(),
         'Konflik label dalam pasien')
    need(int(dev.y_true.eq('0').sum()) == 944 and int(dev.y_true.eq('1').sum()) == 943,
         'Keseimbangan DS944/DR943 tidak sesuai')
    for variant in ('raw','clahe'):
        need(dev[f'{variant}_path'].nunique() == len(dev), f'Path PNG {variant} berulang')

    e5_split_dir = e5_root / 'data' / 'manifests' / 'experiment5_split_70_15_15_seed42_v1'
    e5_png_root = e5_root / 'data' / 'processed' / 'experiment5_letterbox518_v1'
    e5_split_lock = e5_split_dir / 'experiment5_split_70_15_15_seed42_LOCKED.csv'
    e5_proc = e5_png_root / 'processed_manifest.csv'
    e5_test = e5_split_dir / 'test.csv'
    need(checksum(e5_split_lock) == SOURCE_E5_SPLIT_SHA, 'E5 original split lock berubah')
    need(checksum(e5_proc) == SOURCE_E5_PROCESSED_SHA, 'E5 processed manifest berubah')
    need(checksum(e5_test) == SOURCE_TEST_SHA, 'E5 test.csv berubah; holdout harus identik')
    e5s, e5p, test = read_csv(e5_split_lock), read_csv(e5_proc), read_csv(e5_test)
    need(len(test) == 333 and test.patient_id.nunique() == 259,
         'E5 test bukan 333 citra/259 pasien')
    need(len(e5s) == len(e5p) == 2220, 'E5 total bukan 2220')
    original_development = e5s[e5s.split.isin(('train','validation'))].merge(
        e5p[KEY+['raw_path','clahe_path']], on=KEY, how='left', validate='one_to_one')
    need(len(original_development) == len(dev), 'Jumlah development E5 tidak sesuai E6')
    need(same_rows(dev, original_development, ID_COLS),
         'Kandidat fold E6 tidak identik dengan train+validation E5')
    need(not set(dev.patient_id).intersection(set(test.patient_id)),
         'PATIENT LEAKAGE ke fixed test!')
    need(not set(map(tuple,dev[KEY].values)).intersection(set(map(tuple,test[KEY].values))),
         'IMAGE LEAKAGE ke fixed test!')
    counts_in_file = read_csv(lock_dir / 'e6_fold_class_patient_counts.csv')
    observed = []
    folds = {}
    for fold in range(1, N+1):
        pair = {}
        for subset in ('train','validation'):
            raw_file = lock_dir / f'fold_{fold}' / f'{subset}.csv'
            group = read_csv(raw_file)
            need('split' in group.columns and group.split.eq(subset).all(),
                 f'Split label berbeda: Fold {fold} {subset}')
            ref = dev[dev.cv_fold.ne(str(fold)) if subset == 'train' else
                      dev.cv_fold.eq(str(fold))]
            need(len(group) == len(ref) and same_rows(group, ref, list(dev.columns)),
                 f'Fold {fold} {subset} berbeda dari dataset terkunci')
            pair[subset] = group
            ds = group[group.y_true.eq('0')]
            dr = group[group.y_true.eq('1')]
            observed.append({'fold':str(fold),'subset':subset,'images':str(len(group)),
                             'patients':str(group.patient_id.nunique()),
                             'ds_images':str(len(ds)),'dr_images':str(len(dr)),
                             'ds_patients':str(ds.patient_id.nunique()),
                             'dr_patients':str(dr.patient_id.nunique())})
        need(not set(pair['train'].patient_id).intersection(pair['validation'].patient_id),
             f'Patient leakage Fold {fold}')
        need(len(pair['train'])+len(pair['validation']) == 1887,
             f'Fold {fold} tidak menjumlah 1887')
        folds[fold] = pair
    expected_counts = pd.DataFrame(observed)
    need(expected_counts.equals(counts_in_file[expected_counts.columns].astype(str)),
         'Komposisi kelas/pasien tidak cocok dengan ringkasan fold')
    need(sum(len(folds[k]['validation']) for k in folds) == 1887,
         'Cakupan OOF validation tidak lengkap')
    need(set(dev.cv_fold.unique()) == set(('1','2','3','4','5')),
         'Ada fold kosong')
    for variant in ('raw','clahe'):
        for relative in dev[f'{variant}_path']:
            f = png_path(e5_png_root, relative, variant)
            need(f.is_file(), f'PNG development hilang: {f}')
    # Deliberately NEVER open test PNG. Only verify SHA of the fixed test manifest.
    print('INPUT LOCK PASS | E6 1887 development /1471 patients | E5 holdout 333/259 untouched',flush=True)
    for fold in range(1,6):
        tr, va = folds[fold]['train'], folds[fold]['validation']
        print(f'  Fold {fold}: train={len(tr)} ({tr.patient_id.nunique()} patients), '
              f'validation={len(va)} ({va.patient_id.nunique()} patients)',flush=True)
    return dev, folds, report_file, e5_png_root


def make_processor(offline):
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL, local_files_only=offline)
    need(proc.__class__.__name__ == 'BitImageProcessor',
         f'Expected BitImageProcessor, got {type(proc).__name__}')
    need(bool(proc.do_rescale) and bool(proc.do_normalize) and bool(proc.do_convert_rgb),
         'Rescale/normalize/RGB processor tidak aktif')
    need(np.allclose(proc.image_mean,[0.5307]*3, atol=1e-5, rtol=0) and
         np.allclose(proc.image_std,[0.2583]*3, atol=1e-5, rtol=0),
         'Mean/std RAD-DINO berubah')
    need(math.isclose(float(proc.rescale_factor), 1/255, rel_tol=1e-6),
         'Rescale factor RAD-DINO berubah')
    return proc


class CVPNGDataset:
    def __init__(self, rows, root, variant, processor):
        self.rows = rows.sort_values(KEY).reset_index(drop=True)
        self.root, self.variant, self.processor = root, variant, processor
        need(len(self.rows) > 0 and self.rows.split.isin(('train','validation')).all(),
             'Dataset hanya boleh train dan validation E6')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        import torch
        from PIL import Image
        row = self.rows.iloc[index]
        path = png_path(self.root, row[f'{self.variant}_path'], self.variant)
        with Image.open(path) as im:
            need(im.mode == 'L' and im.size == (SIZE,SIZE),
                 f'PNG tidak 8bit grayscale L 518x518: {path}')
            rgb = im.convert('RGB')
            tensor = self.processor(images=rgb, do_resize=False,
                                    do_center_crop=False, return_tensors='pt')['pixel_values'][0]
        need(tuple(tensor.shape) == (3,SIZE,SIZE),f'Tensor size salah: {tuple(tensor.shape)}')
        return {'pixel_values':tensor, 'labels':torch.tensor(int(row['y_true']),dtype=torch.long),
                'patient_id':row['patient_id'],
                'image_key':row['source_archive']+'::'+row['archive_member_path']}


def validate_batch(batch, processor):
    import torch
    x,y = batch['pixel_values'],batch['labels']
    need(x.dtype == torch.float32 and x.ndim == 4 and tuple(x.shape[1:])==(3,SIZE,SIZE),
         f'Wrong image tensor: {tuple(x.shape)} / {x.dtype}')
    need(torch.isfinite(x).all().item(), 'Nonfinite pixel values')
    need(y.dtype == torch.int64 and (bool(((y == 0)|(y == 1)).all().item())),
         'Wrong label tensor')
    need(len(batch['patient_id']) == len(batch['image_key']) == len(y), 'Bad batch metadata')
    need(len(set(batch['image_key'])) == len(y), 'Duplicate image key within batch')
    channel_delta = max(float((x[:,0]-x[:,1]).abs().max()),
                        float((x[:,1]-x[:,2]).abs().max()))
    need(channel_delta <= 1e-5, f'Channels differ unexpectedly: {channel_delta}')
    lower = -float(processor.image_mean[0])/float(processor.image_std[0])
    upper = (1-float(processor.image_mean[0]))/float(processor.image_std[0])
    need(float(x.min()) >= lower-1e-4 and float(x.max()) <= upper+1e-4,
         'Double normalization or invalid pixel range')
    return {'shape':list(x.shape),'n_DS':int(y.eq(0).sum()),
            'n_DR':int(y.eq(1).sum()),'tensor_min':float(x.min()),
            'tensor_max':float(x.max()),'channel_max_delta':channel_delta}


def verify_one_normalization(dataset, batch, processor):
    from PIL import Image
    rel = dataset.rows.iloc[0][f'{dataset.variant}_path']
    with Image.open(png_path(dataset.root, rel, dataset.variant)) as im:
        arr = np.asarray(im,dtype=np.float32)
    expected = (arr*np.float32(processor.rescale_factor)-np.float32(processor.image_mean[0]))/np.float32(processor.image_std[0])
    got = batch['pixel_values'][0,0].cpu().numpy()
    delta = float(np.max(np.abs(expected-got)))
    need(delta <= 5e-5, f'Processor bukan normalisasi tepat sekali, delta={delta}')
    return delta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true', help='Check 5fold locks and image existence only')
    mode.add_argument('--run',action='store_true', help='Run DataLoader smoke for all fold/variante/split')
    ap.add_argument('--batch-size',type=int,default=4, help='Only DataLoader smoke, NOT training batch hyperparameter')
    ap.add_argument('--batches',type=int,default=2)
    ap.add_argument('--offline-processor',action='store_true')
    ap.add_argument('--e6-root',type=Path,default=None)
    ap.add_argument('--e5-root',type=Path,default=None)
    args=ap.parse_args()
    need(args.batch_size>0 and args.batches>0,'batch-size/batches harus positif')
    e6 = (args.e6_root or Path(__file__).resolve().parents[1]).resolve()
    e5 = (args.e5_root or e6.parent/'eksperimen5').resolve()
    need(e6 != e5 and e5 not in e6.parents and e6 not in e5.parents,
         'Root eksperimen6 harus terpisah dari E5')
    dev,folds,lockfile,png_root = load_and_check(e6,e5)
    if args.check:
        print('DRY CHECK PASS_E6_DATALOADER_LOCKS | tidak ada PNG yang dibaca/diubah',flush=True)
        return 0

    import torch
    from torch.utils.data import DataLoader
    processor=make_processor(args.offline_processor)
    print('RAD-DINO PROCESSOR PASS | normalize once | do_resize=False do_center_crop=False',flush=True)
    entries=[]
    for fold in range(1,6):
        for subset in ('train','validation'):
            for variant in ('raw','clahe'):
                dataset=CVPNGDataset(folds[fold][subset],png_root,variant,processor)
                loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,
                                  num_workers=0,pin_memory=False)
                observations=[]
                delta=None
                for ix,batch in enumerate(loader):
                    observations.append(validate_batch(batch,processor))
                    if ix==0:
                        delta=verify_one_normalization(dataset,batch,processor)
                    if ix+1 >= args.batches:
                        break
                need(observations,f'Tidak ada batch untuk Fold {fold}/{subset}/{variant}')
                print(f'  F{fold} {subset:<10} {variant:<5} | batches={len(observations)} '
                      f'| shape={observations[0]["shape"]} | normalization PASS',flush=True)
                entries.append({'fold':fold,'subset':subset,'variant':variant,
                                'n_images':len(dataset),'batches_checked':len(observations),
                                'max_first_image_delta':delta,'batches':observations})
    out=e6/'audits'/'03_dataloader_validation'
    path=out/'03_e6_dataloader_validation_report.json'
    need(not path.exists(),f'Audit report already exists; will not overwrite: {path}')
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS_E6_DATALOADER_VALIDATION','experiment':'experiment6',
            'source_fold_report_sha256':checksum(lockfile),
            'e5_fixed_holdout_test_sha256':SOURCE_TEST_SHA,
            'n_development_images':1887,'n_development_patients':1471,
            'n_holdout_images':333,'n_holdout_patients':259,
            'folds':5,'seed':42,'method':'StratifiedGroupKFold',
            'processor_model':MODEL,'processor_class':type(processor).__name__,
            'torch_version':torch.__version__,
            'normalization':{'rescale_factor':float(processor.rescale_factor),
                             'mean':list(map(float,processor.image_mean)),
                             'std':list(map(float,processor.image_std)),
                             'do_resize':False,'do_center_crop':False,'apply_once':True},
            'smoke_batch_size':args.batch_size,'batches_per_combination':args.batches,
            'n_combinations_checked':len(entries),
            'test_pixel_read':False,'test_inference':False,'model_weights_loaded':False,
            'training_started':False,'e5_modified':False,'e6_folds_modified':False,
            'checks':entries}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'PASS_E6_DATALOADER_VALIDATION | {len(entries)} fold/split/variant combos | report: {path}',flush=True)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'E6 DATALOADER FAIL | {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
