from __future__ import annotations
import argparse
import gc
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
E6_SRC = ROOT.parent / 'eksperimen6' / 'src'
if not (E6_SRC / 'e6_training_common.py').is_file():
    raise RuntimeError(f'Missing E6 utilities at {E6_SRC}')
sys.path.insert(0, str(E6_SRC))
import e6_training_common as C

CONFIG_FILE = Path(__file__).with_name('07_training_config_e7_LOCKED.json')
CONFIG_SHA256 = 'cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'
OUT_BASE = ROOT / 'output' / 'training'
CACHE_BASE = ROOT / 'output' / 'frozen_development_features'


def locked_config():
    C.need(CONFIG_FILE.is_file() and C.sha(CONFIG_FILE) == CONFIG_SHA256,
           'E7 configuration missing or changed')
    cfg = C.read_json(CONFIG_FILE)
    C.need(cfg['status'] == 'LOCKED_E7_FOUR_MODELS', 'Incorrect E7 protocol')
    C.need(cfg['feature_extraction'] == 'CLS' and cfg['classification_head_dropout'] == .3 and
           cfg['learning_rate_head'] == 1e-4 and cfg['weight_decay'] == .01 and
           cfg['scheduler'] == 'CosineAnnealingLR' and cfg['n_training_runs'] == 20,
           'Frozen E7 parameters differ from locked configuration')
    return cfg


def preflight(fold):
    cfg = locked_config()
    _, rows, fp = C.preflight(fold)
    C.need(fp['e6_dev_locked'] == cfg['reference_e6_fold_manifest_sha256'] and
           fp['e5_test_manifest'] == cfg['reference_e5_test_manifest_sha256'] and
           len(rows['train']) + len(rows['validation']) == 1887,
           'E6 five-fold reference changed')
    return cfg, rows, fp


def head_new(cfg):
    return nn.Sequential(nn.Dropout(p=cfg['classification_head_dropout']), nn.Linear(768, 2))


def features_development(variant, processor, device, offline, batch_size, cfg):
    """Cache CLS vectors for 1887 DEVELOPMENT images only; never E5 test."""
    C.need(batch_size >= 1, 'Physical extractor batch must be positive')
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    dest = CACHE_BASE / f'{variant}_1887.pt'
    dev = C.read_csv(C.FOLDS / 'e6_development_1887_5fold_LOCKED.csv').sort_values(C.KEY).reset_index(drop=True)
    C.need(len(dev) == 1887 and dev.patient_id.nunique() == 1471 and
           dev.e5_original_split.isin(('train', 'validation')).all(),
           'Unexpected development feature input')
    keys = (dev.source_archive + '::' + dev.archive_member_path).tolist()
    prov = {
        'e7_config_sha256': CONFIG_SHA256,
        'e6_development_manifest_sha256': C.sha(C.FOLDS / 'e6_development_1887_5fold_LOCKED.csv'),
        'e5_processed_manifest_sha256': C.sha(C.STAGE5 / 'processed_manifest.csv'),
        'e5_fixed_test_manifest_sha256': C.sha(C.TEST_SOURCE),
        'model_revision': cfg['backbone_revision'], 'representation': 'CLS', 'variant': variant,
    }
    if dest.is_file():
        obj = C.torch_load_trusted(dest)
        C.need(obj['provenance'] == prov and obj['keys'] == keys and
               obj['labels'].tolist() == dev.y_true.astype(int).tolist() and
               tuple(obj['features'].shape) == (1887, 768) and
               bool(torch.isfinite(obj['features']).all().item()),
               f'Existing feature cache failed provenance validation: {dest}')
        print(f'PASS_E7_FROZEN_FEATURE_CACHE {variant}', flush=True)
        return obj
    model = C.backbone_new(offline).to(device).eval()
    revision = getattr(model.config, '_commit_hash', None)
    C.need(revision == cfg['backbone_revision'], 'RAD-DINO revision changed')
    for p in model.parameters():
        p.requires_grad_(False)
    dev = dev.copy(); dev['split'] = 'train'
    loader = C.dataset_loader(dev, variant, processor, batch_size, False)
    xs, ys, loaded = [], [], []
    with torch.inference_mode():
        for i, (x, y, sample_keys, _) in enumerate(loader, 1):
            features = model(pixel_values=x.to(device, non_blocking=True)).last_hidden_state[:, 0, :].float().cpu()
            C.need(tuple(features.shape) == (len(y), 768) and
                   bool(torch.isfinite(features).all().item()), 'Nonfinite frozen CLS')
            xs.append(features); ys.append(y); loaded.extend(sample_keys)
            if i % 100 == 0 or i == len(loader):
                print(f'E7 FROZEN CLS {variant}: {i}/{len(loader)} batches', flush=True)
    result = {'features': torch.cat(xs), 'labels': torch.cat(ys), 'keys': loaded,
              'provenance': prov, 'model_commit': revision}
    C.need(loaded == keys and result['labels'].tolist() == dev.y_true.astype(int).tolist(),
           'Frozen feature key/label mismatch')
    C.atomic_torch(dest, result)
    del model, loader
    gc.collect(); torch.cuda.empty_cache()
    return result


def select_features(cache, rows):
    positions = {key: i for i, key in enumerate(cache['keys'])}
    keys = (rows.source_archive + '::' + rows.archive_member_path).tolist()
    C.need(len(keys) == len(set(keys)) and all(key in positions for key in keys),
           'Fold not contained in 1887 development feature cache')
    ids = [positions[key] for key in keys]
    x, y = cache['features'][ids], cache['labels'][ids]
    C.need(y.tolist() == rows.y_true.astype(int).tolist(), 'Feature labels mismatch')
    return x, y


def evaluate_head(head, x, y, device):
    head.eval()  # Dropout OFF for train/validation evaluation
    loss_sum = 0.0; correct = 0; probs = []
    with torch.inference_mode():
        for i in range(0, len(y), 256):
            logits = head(x[i:i+256].to(device)).float()
            target = y[i:i+256].to(device)
            loss_sum += float(nn.functional.cross_entropy(logits, target, reduction='sum').item())
            correct += int((logits.argmax(1) == target).sum().item())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return loss_sum / len(y), correct / len(y), np.asarray(probs, dtype=float)


def main(args):
    cfg, rows, fp = preflight(args.fold)
    C.need(args.feature_batch_size >= 1 and not (args.resume and args.smoke),
           'Invalid batch or smoke/resume combination')
    if args.check:
        print(f'PASS_E7_PREFLIGHT | frozen/{args.variant}/fold{args.fold} | '
              f'train={len(rows["train"])} validation={len(rows["validation"])} | test untouched', flush=True)
        return
    C.seed_all(cfg['model_seed'])
    device = C.cuda_device()
    if args.smoke:
        processor = C.processor_new(args.offline_model)
        model = C.backbone_new(args.offline_model).to(device).eval()
        for p in model.parameters():p.requires_grad_(False)
        x, y, _, _ = next(iter(C.dataset_loader(rows['train'].iloc[:1], args.variant, processor, 1, False)))
        with torch.no_grad():
            feat = model(pixel_values=x.to(device)).last_hidden_state[:, 0, :].float().clone().detach()
        head = head_new(cfg).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate_head'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler_T_max_epochs'], eta_min=cfg['scheduler_eta_min'])
        loss = nn.functional.cross_entropy(head(feat), y.to(device))
        C.need(bool(torch.isfinite(loss).item()), 'Nonfinite Frozen E7 smoke loss')
        loss.backward(); optimizer.step(); scheduler.step()
        C.need(scheduler.last_epoch == 1, 'Frozen E7 scheduler smoke failed')
        print(f'PASS_E7_GPU_SMOKE | frozen {args.variant} fold{args.fold} | '
              f'loss={loss.item():.5f} | peak_allocated_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}', flush=True)
        return

    out = OUT_BASE / f'frozen_{args.variant}_fold_{args.fold}'
    if args.resume:
        C.need(out.is_dir() and (out/'latest_training_state.pt').is_file() and
               not (out/'run_summary.json').exists(), 'No unfinished Frozen E7 checkpoint')
    else:
        C.need(not out.exists(), f'Existing output will not be overwritten: {out}')
        out.mkdir(parents=True, exist_ok=False)
    processor = C.processor_new(args.offline_model)
    cache = features_development(args.variant, processor, device, args.offline_model, args.feature_batch_size, cfg)
    x, y = select_features(cache, rows['train'])
    vx, vy = select_features(cache, rows['validation'])
    head = head_new(cfg).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate_head'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler_T_max_epochs'], eta_min=cfg['scheduler_eta_min'])
    run_cfg = {'experiment': 'experiment7', 'model': f'frozen_{args.variant}', 'fold': args.fold,
               'config_sha256': CONFIG_SHA256, 'fold_source_fingerprints': fp,
               'feature_cache_provenance': cache['provenance'], 'model_commit': cache['model_commit'],
               'feature_extractor_batch': args.feature_batch_size, 'head_batch': cfg['frozen_head_batch_size'],
               'test_read': False, 'config': cfg}
    if args.resume:
        C.need(C.read_json(out/'run_config.json') == run_cfg, 'Resume config mismatch')
    else:
        C.atomic_json(out/'run_config.json', run_cfg)
    last = 0; best = float('-inf'); best_epoch = 0; stale = 0; history = []
    if args.resume:
        saved = C.torch_load_trusted(out/'latest_training_state.pt')
        C.need(saved['experiment'] == 'experiment7' and saved['model'] == f'frozen_{args.variant}' and
               saved['fold'] == args.fold and saved['config_sha256'] == CONFIG_SHA256 and
               saved['fold_source_fingerprints'] == fp and saved['feature_cache_provenance'] == cache['provenance'],
               'Frozen resume provenance mismatch')
        head.load_state_dict(saved['head_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        C.restore_rng(saved['rng_state'])
        last = int(saved['epoch']); best = float(saved['best_validation_auroc'])
        best_epoch = int(saved['best_epoch']); stale = int(saved['stale_epochs']); history = saved['history']
        C.need(last == len(history) and scheduler.last_epoch == last,
               'Frozen resume epoch / scheduler / history mismatch')
        print(f'RESUMED E7 FROZEN {args.variant} fold{args.fold} after epoch {last}', flush=True)
    print(f'START E7 FROZEN {args.variant} fold{args.fold} | train={len(x)} validation={len(vx)}'
          f' head_batch={cfg["frozen_head_batch_size"]}', flush=True)
    for epoch in range(last + 1, cfg['max_epochs'] + 1) if stale < cfg['early_stopping_patience'] else []:
        head.train()
        g = torch.Generator().manual_seed(cfg['model_seed'] + epoch)
        dl = DataLoader(TensorDataset(x, y), batch_size=cfg['frozen_head_batch_size'],
                        shuffle=True, generator=g, drop_last=False, num_workers=0)
        epoch_lr = optimizer.param_groups[0]['lr']
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(xb), yb)
            C.need(bool(torch.isfinite(loss).item()), f'Nonfinite Frozen training epoch {epoch}')
            loss.backward(); optimizer.step()
        train_loss, train_accuracy, _ = evaluate_head(head, x, y, device)
        val_loss, val_accuracy, val_p = evaluate_head(head, vx, vy, device)
        metrics = C.classification_metrics(vy.numpy(), val_p)
        better = metrics['auroc'] > best + cfg['early_stopping_min_delta']
        if better:
            best = metrics['auroc']; best_epoch = epoch; stale = 0
            C.atomic_torch(out/'best_classifier.pt', {
                'classifier_state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                'best_epoch': epoch, 'model': f'frozen_{args.variant}', 'fold': args.fold,
                'config_sha256': CONFIG_SHA256, 'source_fingerprints': fp, 'model_commit': cache['model_commit'],
                'head': 'Dropout(0.3)+Linear(768,2)', 'feature': 'CLS'})
            C.atomic_csv(out/'best_validation_predictions.csv', C.validation_prediction_csv(rows['validation'], val_p))
            C.atomic_json(out/'best_validation_metrics.json', {'best_epoch': epoch, **metrics})
        else:
            stale += 1
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                        'val_loss': val_loss, 'val_accuracy': val_accuracy,
                        'val_auroc': metrics['auroc'], 'val_macro_f1': metrics['macro_f1'],
                        'val_balanced_accuracy': metrics['balanced_accuracy'],
                        'val_sensitivity_dr': metrics['sensitivity_dr'],
                        'val_specificity_ds': metrics['specificity_ds'],
                        'lr_head': epoch_lr, 'optimizer_steps': len(dl)})
        C.atomic_csv(out/'training_history.csv', pd.DataFrame(history))
        C.atomic_torch(out/'latest_training_state.pt', {
            'experiment': 'experiment7', 'model': f'frozen_{args.variant}', 'fold': args.fold,
            'config_sha256': CONFIG_SHA256, 'fold_source_fingerprints': fp,
            'feature_cache_provenance': cache['provenance'], 'epoch': epoch,
            'best_validation_auroc': best, 'best_epoch': best_epoch, 'stale_epochs': stale,
            'history': history, 'head_state': {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'rng_state': C.rng_state()})
        print(f'E7 FROZEN {args.variant} F{args.fold} EPOCH {epoch:02d} | '
              f'train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} val_AUROC={metrics["auroc"]:.4f} '
              f'| {"BEST" if better else "no_improve="+str(stale)}', flush=True)
        if stale >= cfg['early_stopping_patience']:
            print(f'Early stopping Frozen {args.variant} fold{args.fold} epoch{epoch}', flush=True)
            break
    C.need(best_epoch > 0 and (out/'best_classifier.pt').is_file() and
           (out/'best_validation_predictions.csv').is_file() and len(history),
           'Frozen E7 missing best checkpoint or validation predictions')
    C.atomic_json(out/'run_summary.json', {
        'status': 'PASS_E7_FOLD_TRAINING_COMPLETE', 'experiment': 'experiment7',
        'model': f'frozen_{args.variant}', 'method': 'frozen', 'variant': args.variant,
        'fold': args.fold, 'best_epoch': best_epoch, 'best_validation_auroc': best,
        'last_epoch': history[-1]['epoch'], 'epochs_recorded': len(history),
        'train_images': len(x), 'validation_images': len(vx),
        'source_e6_fingerprints': fp, 'locked_config_sha256': CONFIG_SHA256,
        'model_commit': cache['model_commit'], 'test_read': False, 'training_only_development': True})
    print(f'PASS_E7_FOLD_TRAINING_COMPLETE | frozen {args.variant} fold {args.fold} '
          f'best_epoch={best_epoch} val_AUROC={best:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', required=True, type=int, choices=range(1, 6))
    parser.add_argument('--variant', required=True, choices=('raw', 'clahe'))
    parser.add_argument('--feature-batch-size', type=int, default=1,
                        help='Frozen backbone extraction batch; head batch size is 64')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--offline-model', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        print(f'FAIL_E7_FROZEN | {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
