from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# Reference E6 library is reused read-only; it checks E5/E6 checksum and patient leakage.
E7_ROOT = Path(__file__).resolve().parents[1]
E6_SRC = E7_ROOT.parent / 'eksperimen6' / 'src'
if not (E6_SRC / 'e6_training_common.py').is_file():
    raise RuntimeError(f'Missing E6 shared utilities; leave Eksperimen 6 unchanged: {E6_SRC}')
sys.path.insert(0, str(E6_SRC))

import numpy as np
import pandas as pd
import torch
from torch import nn
import e6_training_common as C

CONFIG_PATH = Path(__file__).with_name('07_training_config_e7_LOCKED.json')
CONFIG_SHA256 = 'cd0fcdfd7cab476c2408041d2966b5171d85421e9d46d90af299a3c9dd69fd05'
OUT_BASE = E7_ROOT / 'output' / 'training'


def load_e7_config():
    C.need(CONFIG_PATH.is_file() and C.sha(CONFIG_PATH) == CONFIG_SHA256,
           'E7 locked configuration missing or modified')
    d = C.read_json(CONFIG_PATH)
    C.need(d['status'] == 'LOCKED_E7_FOUR_MODELS', 'Invalid E7 lock')
    C.need(len(d.get('model_variants', [])) == 4 and
           d['learning_rate_head'] == 1e-4 and d['learning_rate_lora'] == 1e-4 and
           d['weight_decay'] == .01 and d['lora_rank'] == 8 and d['lora_alpha'] == 8 and
           d['classification_head_dropout'] == .3 and d['feature_extraction'] == 'CLS',
           'Incorrect E7 four-model hyperparameters')
    return d


def preflight(fold):
    cfg = load_e7_config()
    # Reuse all E6 audit/fold tests rather than re-auditing DICOM or re-splitting.
    _, rows, evidence = C.preflight(fold)
    C.need(evidence['e6_dev_locked'] == cfg['reference_e6_fold_manifest_sha256'],
           'E6 fold dataset changed')
    C.need(evidence['e5_test_manifest'] == cfg['reference_e5_test_manifest_sha256'],
           'E5 fixed holdout manifest changed')
    C.need(len(rows['train']) + len(rows['validation']) == 1887,
           'Unexpected fold size')
    C.need(all(r['split'].eq(subset).all() for subset, r in rows.items()),
           'Unexpected train/validation split')
    return cfg, rows, evidence


def make_lora(backbone, cfg):
    from peft import LoraConfig, get_peft_model
    adapter_config = LoraConfig(
        r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['lora_target_modules'],
        layers_to_transform=cfg['lora_layers_zero_based'],
        layers_pattern='layer', bias='none')
    model = get_peft_model(backbone, adapter_config)
    trained = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    C.need(len(trained) > 0 and all('lora_A' in name or 'lora_B' in name for name, _ in trained),
           'Unexpected LoRA trainable parameter names')
    C.need(sum(p.numel() for _, p in trained) == 98304,
           'Unexpected number of LoRA trainable parameters')
    return model


def make_head(cfg):
    head = nn.Sequential(nn.Dropout(p=cfg['classification_head_dropout']), nn.Linear(768, 2))
    C.need(sum(p.numel() for p in head.parameters()) == 1538, 'Unexpected head size')
    return head


def copy_trainable_state(backbone, saved):
    ps = {k: p for k, p in backbone.named_parameters() if p.requires_grad}
    C.need(set(ps) == set(saved), 'Changed LoRA parameter names; resume refused')
    with torch.no_grad():
        for k, p in ps.items():
            p.copy_(saved[k].to(device=p.device, dtype=p.dtype))


def stats_and_predictions(backbone, head, rows, processor, device, variant):
    backbone.eval(); head.eval()
    dl = C.dataset_loader(rows, variant, processor, 1, False)
    labels, keys, probs = [], [], []
    total_loss = 0.0; correct = 0
    with torch.inference_mode():
        for x, y, key, _ in dl:
            with torch.cuda.amp.autocast(enabled=True):
                features = backbone(pixel_values=x.to(device, non_blocking=True)).last_hidden_state[:, 0, :]
                logits = head(features)
            logits = logits.float()
            target = y.to(device)
            loss = nn.functional.cross_entropy(logits, target, reduction='sum')
            C.need(bool(torch.isfinite(loss).item()), 'Validation loss nonfinite')
            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == target).sum().item())
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(y.tolist()); keys.extend(key)
    expected = (rows.source_archive + '::' + rows.archive_member_path).tolist()
    C.need(keys == expected and labels == rows.y_true.astype(int).tolist(),
           'Validation order/labels inconsistent with locked manifest')
    metrics = C.classification_metrics(labels, probs)
    return total_loss / len(rows), correct / len(rows), np.asarray(probs), metrics


def archive_leftover(path, run_dir):
    """Never delete a previous attempted export. Move it aside if redoing an uncommitted epoch."""
    if not path.exists():
        return
    recover = run_dir / 'recovered_incomplete_exports'
    recover.mkdir(exist_ok=True)
    i = 1
    while (recover / f'{path.name}.recovery_{i:03d}').exists():
        i += 1
    path.rename(recover / f'{path.name}.recovery_{i:03d}')


def save_best_artifacts(backbone, head, rows, probs, metrics, ep, out, evidence, revision):
    """Immutable per-best-epoch paths avoid E6's best_adapter_staging/backup collision."""
    folder = out / 'best_candidates'
    folder.mkdir(exist_ok=True)
    name = f'epoch_{ep:02d}'
    target = folder / name
    stage = folder / (name + '.staging')
    # This epoch cannot be committed if the previous complete state is < ep.
    archive_leftover(stage, out)
    archive_leftover(target, out)
    backbone.save_pretrained(stage, safe_serialization=True)
    C.need((stage/'adapter_config.json').is_file() and
           (stage/'adapter_model.safetensors').is_file() and
           (stage/'adapter_model.safetensors').stat().st_size > 0,
           'Incomplete adapter export')
    stage.rename(target)
    C.atomic_torch(target/'best_classifier.pt', {
        'classifier_state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        'best_epoch': ep, 'fold': evidence['fold'],
        'model_commit': revision, 'fingerprints': evidence,
        'feature': 'CLS', 'head': 'Dropout(0.3)+Linear(768,2)'})
    C.atomic_csv(target/'best_validation_predictions.csv',
                 C.validation_prediction_csv(rows, probs))
    C.atomic_json(target/'best_validation_metrics.json',
                  {'best_epoch': ep, **metrics})
    return str(target.relative_to(out))


def set_best_pointer(out, snapshot, epoch, auroc):
    C.need((out/snapshot/'adapter_model.safetensors').is_file(),
           'Cannot point to nonexistent adapter')
    C.need((out/snapshot/'best_classifier.pt').is_file(),
           'Cannot point to nonexistent classification head')
    C.atomic_json(out/'best_checkpoint.json', {
        'epoch': epoch, 'validation_auroc': auroc,
        'snapshot_relative_path': snapshot,
        'adapter_path': snapshot,
        'classifier_path': f'{snapshot}/best_classifier.pt',
        'predictions_path': f'{snapshot}/best_validation_predictions.csv',
        'metrics_path': f'{snapshot}/best_validation_metrics.json'})


def make_optimizer(backbone, head, cfg):
    params = [p for p in backbone.parameters() if p.requires_grad]
    return torch.optim.AdamW([
        {'params': params, 'lr': cfg['learning_rate_lora'], 'group_name': 'lora'},
        {'params': list(head.parameters()), 'lr': cfg['learning_rate_head'], 'group_name': 'head'}
    ], weight_decay=cfg['weight_decay'])


def make_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['scheduler_T_max_epochs'], eta_min=cfg['scheduler_eta_min'])


def smoke(backbone, head, processor, device, rows, cfg, variant):
    backbone.train(); head.train()
    optimizer = make_optimizer(backbone, head, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)
    dl = C.dataset_loader(rows.iloc[:16], variant, processor, 1, False)
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y, _, _) in enumerate(dl, 1):
        with torch.cuda.amp.autocast(enabled=True):
            cls = backbone(pixel_values=x.to(device)).last_hidden_state[:, 0, :]
            logits = head(cls)
            loss = nn.functional.cross_entropy(logits.float(), y.to(device))
        C.need(bool(torch.isfinite(loss).item()), f'Nonfinite smoke loss image {i}')
        scaler.scale(loss / 16).backward()
    adapter_ps = [p for p in backbone.parameters() if p.requires_grad]
    C.need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in adapter_ps),
           'LoRA gradients missing/invalid')
    C.need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in head.parameters()),
           'Head gradients missing/invalid')
    before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
    C.need(scaler.get_scale() >= before, 'AMP skipped smoke optimizer step')
    scheduler.step()
    C.need(scheduler.last_epoch == 1 and all(g['lr'] > 0 for g in optimizer.param_groups),
           'Cosine scheduler failed')
    print(f'PASS_E7_GPU_SMOKE | LoRA {variant} + CLS + HeadDropout0.3 + CosineAnnealingLR'
          f' | peak_allocated_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}', flush=True)


def main(a):
    cfg, rows, evidence = preflight(a.fold)
    out = OUT_BASE / f'lora_{a.variant}_fold_{a.fold}'
    if a.check:
        print(f'PASS_E7_PREFLIGHT | fold={a.fold} train={len(rows["train"])} '
              f'validation={len(rows["validation"])} | E5/E6 source files unchanged', flush=True)
        return
    C.need(not (a.resume and a.smoke), 'Do not combine --resume with --smoke')
    C.seed_all(cfg['model_seed'])
    device = C.cuda_device()
    if not a.smoke:
        if a.resume:
            C.need(out.is_dir() and (out/'latest_training_state.pt').is_file() and
                   not (out/'run_summary.json').exists(),
                   'Resume requires unfinished E7 run and latest_training_state.pt')
        else:
            C.need(not out.exists(), f'Run output already exists (no overwriting): {out}')
            out.mkdir(parents=True)
    processor = C.processor_new(a.offline_model)
    backbone = C.backbone_new(a.offline_model)
    revision = getattr(backbone.config, '_commit_hash', None)
    C.need(revision == cfg['backbone_revision'], 'Unexpected RAD-DINO revision')
    if a.gradient_checkpointing:
        C.need(hasattr(backbone, 'gradient_checkpointing_enable'),
               'Backbone gradient checkpointing unsupported')
        backbone.gradient_checkpointing_enable()
    backbone = make_lora(backbone, cfg).to(device)
    head = make_head(cfg).to(device)
    if a.smoke:
        smoke(backbone, head, processor, device, rows['train'], cfg, a.variant)
        return

    run_cfg = {
        'experiment': 'experiment7', 'method': 'lora', 'variant': a.variant, 'fold': a.fold,
        'locked_config_sha256': CONFIG_SHA256,
        'source_e6_fingerprints': evidence, 'model_commit': revision,
        'gradient_checkpointing': bool(a.gradient_checkpointing),
        'configuration': cfg, 'test_read': False}
    if a.resume:
        C.need(C.read_json(out/'run_config.json') == run_cfg,
               'Run config changed; refuse resume')
    else:
        C.atomic_json(out/'run_config.json', run_cfg)
    optimizer = make_optimizer(backbone, head, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)
    last = 0; best = float('-inf'); best_ep = 0; stale = 0; history = []
    best_snapshot = None
    if a.resume:
        saved = C.torch_load_trusted(out/'latest_training_state.pt')
        C.need(saved['experiment'] == 'experiment7' and saved['fold'] == a.fold and
               saved.get('variant') == a.variant and
               saved['source_e6_fingerprints'] == evidence and
               saved['locked_config_sha256'] == CONFIG_SHA256 and
               saved['model_commit'] == revision and
               saved['gradient_checkpointing'] == bool(a.gradient_checkpointing),
               'Resume provenance mismatch')
        copy_trainable_state(backbone, saved['lora_state'])
        head.load_state_dict(saved['head_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        scaler.load_state_dict(saved['scaler_state'])
        C.restore_rng(saved['rng_state'])
        last = int(saved['epoch']); best = float(saved['best_auroc'])
        best_ep = int(saved['best_epoch']); stale = int(saved['stale_epochs'])
        history = saved['history']; best_snapshot = saved['best_snapshot_relative_path']
        C.need(last == len(history) and scheduler.last_epoch == last,
               'Epoch/scheduler/history mismatch on resume')
        if best_snapshot:
            set_best_pointer(out, best_snapshot, best_ep, best)
        print(f'RESUMED E7 LoRA {a.variant} fold {a.fold} after epoch {last}', flush=True)
    print(f'START E7 LoRA {a.variant} fold={a.fold} train={len(rows["train"])} '
          f'validation={len(rows["validation"])} micro_batch=1 accumulation=16', flush=True)
    if stale >= cfg['early_stopping_patience']:
        print(f'Resume state already satisfies early stopping: epoch {last}', flush=True)
    for ep in range(last + 1, cfg['max_epochs'] + 1) if stale < cfg['early_stopping_patience'] else []:
        backbone.train(); head.train()
        loader = C.dataset_loader(rows['train'], a.variant, processor, 1, True, ep)
        count = len(loader); train_loss = 0.0; train_right = 0; steps = 0; skips = 0
        lr_lora, lr_head = (g['lr'] for g in optimizer.param_groups)
        optimizer.zero_grad(set_to_none=True)
        for i, (x, y, _, _) in enumerate(loader):
            group_start = (i // 16) * 16
            group_size = min(16, count - group_start)
            xb = x.to(device, non_blocking=True); yb = y.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                cls = backbone(pixel_values=xb).last_hidden_state[:, 0, :]
                logits = head(cls)
                loss = nn.functional.cross_entropy(logits.float(), yb)
            C.need(bool(torch.isfinite(loss).item()),
                   f'Nonfinite training loss fold {a.fold} ep{ep} micro-batch {i+1}')
            scaler.scale(loss / group_size).backward()
            train_loss += float(loss.item())
            train_right += int((logits.detach().float().argmax(dim=1) == yb).sum().item())
            if (i+1) % 16 == 0 or i+1 == count:
                if ep == 1 and i+1 == 16:
                    C.need(all(p.grad is not None and torch.isfinite(p.grad).all().item()
                               for p in backbone.parameters() if p.requires_grad),
                           'Missing LoRA gradients')
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < previous_scale:
                    skips += 1
                else:
                    steps += 1
                optimizer.zero_grad(set_to_none=True)
            if (i+1) % 200 == 0 or i+1 == count:
                print(f'  E7 {a.variant} F{a.fold} epoch {ep:02d}: '
                      f'micro-batch {i+1}/{count} | steps={steps} | amp_skips={skips}',
                      flush=True)
        C.need(skips <= 3, f'Too many AMP skipped steps: {skips}')
        val_loss, val_acc, probs, metrics = stats_and_predictions(
            backbone, head, rows['validation'], processor, device, a.variant)
        better = metrics['auroc'] > best + cfg['early_stopping_min_delta']
        snapshot = best_snapshot
        if better:
            snapshot = save_best_artifacts(backbone, head, rows['validation'], probs,
                                           metrics, ep, out, evidence, revision)
            best = metrics['auroc']; best_ep = ep; stale = 0
        else:
            stale += 1
        # Schedule advances once per *completed epoch*, including non-improvements.
        scheduler.step()
        row = {
            'epoch': ep, 'train_loss': train_loss/count, 'train_accuracy': train_right/count,
            'val_loss': val_loss, 'val_accuracy': val_acc,
            'val_auroc': metrics['auroc'], 'val_macro_f1': metrics['macro_f1'],
            'val_balanced_accuracy': metrics['balanced_accuracy'],
            'val_sensitivity_dr': metrics['sensitivity_dr'],
            'val_specificity_ds': metrics['specificity_ds'],
            'lr_lora': lr_lora, 'lr_head': lr_head,
            'optimizer_steps': steps, 'fp16_skipped_steps': skips}
        history.append(row)
        C.atomic_csv(out/'training_history.csv', pd.DataFrame(history))
        # Commit epoch and scheduler/optimizer together. The pointer is reconstructable.
        C.atomic_torch(out/'latest_training_state.pt', {
            'experiment': 'experiment7', 'fold': a.fold, 'variant': a.variant, 'model_commit': revision,
            'gradient_checkpointing': bool(a.gradient_checkpointing),
            'source_e6_fingerprints': evidence, 'locked_config_sha256': CONFIG_SHA256,
            'epoch': ep, 'best_auroc': best, 'best_epoch': best_ep,
            'stale_epochs': stale, 'history': history,
            'best_snapshot_relative_path': snapshot,
            'lora_state': {k:p.detach().cpu().clone() for k,p in backbone.named_parameters()
                           if p.requires_grad},
            'head_state': {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'rng_state': C.rng_state()})
        best_snapshot = snapshot
        if best_snapshot:
            set_best_pointer(out, best_snapshot, best_ep, best)
        print(f'E7 {a.variant} F{a.fold} EPOCH {ep:02d} | '
              f'train_loss={train_loss/count:.4f} train_acc={train_right/count:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} '
              f'val_AUROC={metrics["auroc"]:.4f} | '
              f'{"BEST" if better else "no_improve="+str(stale)}', flush=True)
        if stale >= cfg['early_stopping_patience']:
            print(f'Early stopping fold {a.fold} at epoch {ep}', flush=True)
            break
    C.need(best_snapshot is not None and best_ep > 0, 'No best checkpoint available')
    set_best_pointer(out, best_snapshot, best_ep, best)
    pointer = C.read_json(out/'best_checkpoint.json')
    C.need(len(history) > 0 and pointer['epoch'] == best_ep,
           'Run cannot be marked complete without valid checkpoint')
    C.atomic_json(out/'run_summary.json', {
        'status': 'PASS_E7_FOLD_TRAINING_COMPLETE', 'experiment': 'experiment7',
        'model': f'lora_{a.variant}', 'fold': a.fold, 'best_epoch': best_ep,
        'best_validation_auroc': best, 'last_epoch': history[-1]['epoch'],
        'epochs_recorded': len(history),
        'best_snapshot_relative_path': best_snapshot,
        'train_images': len(rows['train']),
        'validation_images': len(rows['validation']),
        'source_e6_fingerprints': evidence,
        'locked_config_sha256': CONFIG_SHA256,
        'model_commit': revision, 'test_read': False, 'training_only_development': True})
    print(f'PASS_E7_FOLD_TRAINING_COMPLETE | fold={a.fold} '
          f'best_epoch={best_ep} best_val_AUROC={best:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', required=True, type=int, choices=range(1,6))
    parser.add_argument('--variant', required=True, choices=('raw','clahe'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--offline-model', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Use consistently from first run through resume')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        print(f'FAIL_E7_TRAINING | {type(error).__name__}: {error}', file=sys.stderr)
        raise
