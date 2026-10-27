"""Fine-tune a time-aware acoustic foundation with circular and spatial losses.

The run reuses the immutable large-train cache and independent heard/unheard
validation episodes prepared for earlier AV-WAN studies. It does not copy the
large cache or touch existing checkpoints.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.avwan.data_pipeline.dataset import make_loader
from experiment.avwan.model.hybrid import Network
from experiment.avwan.model.maps import unpack_geometry
from experiment.avwan.training.trainer import save_checkpoint


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOUNDATION = ROOT / 'result/2026-10-13-staged-grl-reproduction/foundation/action_stage/best.pt'
DEFAULT_TRAIN_CACHE = ROOT / 'result/2026-10-13-staged-grl-reproduction/prepared_data/dataset'
DEFAULT_UNHEARD_CACHE = ROOT / 'result/10-11/total/2026-10-03-unheard-generalization-study/domain_cache'
DEFAULT_OUTPUT = ROOT / 'result/2026-10-17-temporal-audio-foundation'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build_composite_cache(output, train_cache, unheard_cache):
    """Reference immutable train/heard/unheard shards through one manifest."""
    train_manifest = json.loads((train_cache / 'manifest.json').read_text())
    unheard_manifest = json.loads((unheard_cache / 'manifest.json').read_text())
    if (train_manifest['identity']['purpose'] != 'avwan_foundation_spatial' or
            unheard_manifest['identity']['purpose'] != 'avwan_foundation_spatial'):
        raise ValueError('Expected AV-WAN foundation feature caches')
    if (train_manifest['observation_schema'] != unheard_manifest['observation_schema'] or
            train_manifest['action_count'] != unheard_manifest['action_count']):
        raise ValueError('Training and validation cache schemas differ')
    cache = output / 'composite_cache'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = deepcopy(train_manifest)
    for split in ('train', 'val_heard'):
        manifest['splits'][split] = [
            dict(shard, path=str((train_cache / shard['path']).resolve()))
            for shard in train_manifest['splits'][split]]
    manifest['splits']['val_unheard'] = [
        dict(shard, path=str((unheard_cache / shard['path']).resolve()))
        for shard in unheard_manifest['splits']['val']]
    manifest['samples']['val_unheard'] = unheard_manifest['samples']['val']
    manifest['independent_unheard_validation'] = unheard_manifest['independent_unheard_validation']
    manifest['composite_sources'] = {
        'train_and_heard': str(train_cache.resolve()),
        'unheard_validation': str(unheard_cache.resolve()),
    }
    target = cache / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError(f'Composite manifest changed: {target}')
    if not target.exists():
        write_json(target, manifest)
    return cache, manifest


def loaders(cache, batch_size, workers, seed):
    config = {
        'seed': seed,
        'data': {'cache': str(cache)},
        'training': {'batch_size': batch_size, 'workers': workers,
                     'pin_memory': True, 'prefetch_factor': 2},
    }
    return {split: make_loader(config, split)
            for split in ('train', 'val_heard', 'val_unheard')}


def move_batch(batch, device, action_count):
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    rgb = torch.cat((moved['prev_rgb'], moved['rgb']), dim=-1).float() / 255.
    depth = torch.cat((moved['prev_depth'], moved['depth']), dim=-1).float()
    geometry = unpack_geometry(moved['gm'])
    if geometry.ndim == 3:
        geometry = geometry.unsqueeze(0)
    if moved['action_mask'].shape[-1] != action_count:
        raise ValueError('Action mask does not match checkpoint action count')
    return moved, rgb, depth, geometry


def angle_targets(labels):
    # Eight sectors cover [-pi, pi); target the center of each sector.
    angle = labels.float() * (torch.pi / 4.) - 7. * torch.pi / 8.
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1), angle


def spatial_soft_targets(action, mask, sigma=0.85):
    """Gaussian target over legal adjacent 9x9 waypoints; STOP stays exact."""
    count = mask.shape[-1]
    if count != 81:
        raise ValueError(f'Spatial action loss expects 81 actions, got {count}')
    ids = torch.arange(count, device=action.device)
    target_row, target_col = action // 9, action % 9
    row, col = ids // 9, ids % 9
    distance2 = (row[None, :] - target_row[:, None]).square() + (col[None, :] - target_col[:, None]).square()
    weights = torch.exp(-distance2.float() / (2. * sigma * sigma))
    weights = weights * mask.float()
    weights[:, 40] = 0.  # Never blur the STOP class into nearby waypoints.
    moving = action != 40
    weights[~moving] = 0.
    weights[~moving, action[~moving]] = 1.
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights, moving


def metric_sums(model, loader, update, epoch, device, optimizer=None, scaler=None,
                max_steps=0):
    model.train(update)
    if not update:
        model.eval()
    loader.dataset.set_epoch(epoch)
    totals = {}
    seen = 0
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        moved, rgb, depth, geometry = move_batch(batch, device, model.action_count)
        action = moved['action'].long()
        mask = moved['action_mask'].bool()
        sector = moved['angle_label'].long()
        valid_angle = action != model.action_count // 2
        with torch.set_grad_enabled(update), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, angle_logits, _, audio_token = model.forward_joint_features(
                moved['spectrogram'], rgb, depth, geometry, moved['am'], mask,
                return_audio=True)
            logits = logits.float()
            angle_logits = angle_logits.float()
            vector = model.audio_encoder.predict_direction(audio_token)
            unit_vector = F.normalize(vector.float(), dim=-1, eps=1e-6)
            target_vector, target_angle = angle_targets(sector)
            action_loss = F.cross_entropy(logits, action)
            soft_target, moving = spatial_soft_targets(action, mask)
            soft_action_loss = -(soft_target * F.log_softmax(logits, dim=-1)).sum(-1)
            soft_action_loss = soft_action_loss[moving].mean() if moving.any() else action_loss * 0.
            angle_loss = (F.cross_entropy(angle_logits[valid_angle], sector[valid_angle])
                          if valid_angle.any() else angle_logits.sum() * 0.)
            direction_loss = ((1. - (unit_vector * target_vector).sum(-1))[valid_angle].mean()
                              if valid_angle.any() else vector.sum() * 0.)
            loss = action_loss + .2 * soft_action_loss + .15 * angle_loss + .15 * direction_loss

        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite {"train" if update else "validation"} objective')
        if update:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            scaler.step(optimizer)
            scaler.update()

        prediction = logits.argmax(-1)
        angle_prediction = angle_logits.argmax(-1)
        predicted_angle = torch.atan2(unit_vector[:, 1], unit_vector[:, 0])
        circular_error = torch.atan2(torch.sin(predicted_angle - target_angle),
                                     torch.cos(predicted_angle - target_angle)).abs()
        sector_centers = angle_targets(torch.arange(8, device=device))[1]
        legacy_angle = sector_centers[angle_prediction]
        legacy_error = torch.atan2(torch.sin(legacy_angle - target_angle),
                                   torch.cos(legacy_angle - target_angle)).abs()
        moving_prediction = prediction != 40
        local = ((prediction // 9 - action // 9).abs() <= 1) & ((prediction % 9 - action % 9).abs() <= 1)
        n = len(action)
        seen += n
        batch_metrics = {
            'loss': loss,
            'action_loss': action_loss,
            'spatial_action_loss': soft_action_loss,
            'angle_loss': angle_loss,
            'direction_loss': direction_loss,
            'action_accuracy': (prediction == action).float().mean(),
            'moving_action_accuracy': ((prediction == action) & (action != 40)).sum() /
                                      (action != 40).sum().clamp_min(1),
            'neighbor_action_accuracy': (local & (action != 40)).sum() /
                                        (action != 40).sum().clamp_min(1),
            'angle_accuracy': ((angle_prediction == sector) & valid_angle).sum() /
                              valid_angle.sum().clamp_min(1),
            'direction_mae_degrees': circular_error[valid_angle].mean() * (180. / torch.pi)
                                     if valid_angle.any() else circular_error.sum() * 0.,
            'sector_mae_degrees': legacy_error[valid_angle].mean() * (180. / torch.pi)
                                  if valid_angle.any() else legacy_error.sum() * 0.,
        }
        for name, value in batch_metrics.items():
            denominator = n
            if name in ('spatial_action_loss', 'neighbor_action_accuracy'):
                denominator = int(moving.sum())
            elif name in ('angle_loss', 'direction_loss', 'angle_accuracy',
                          'direction_mae_degrees', 'sector_mae_degrees'):
                denominator = int(valid_angle.sum())
            totals[name] = totals.get(name, 0.) + float(value.detach()) * denominator
        totals['direction_samples'] = totals.get('direction_samples', 0.) + int(valid_angle.sum())
        totals['moving_samples'] = totals.get('moving_samples', 0.) + int(moving.sum())
        if max_steps and step >= max_steps:
            break
    if seen == 0:
        raise RuntimeError('Empty data split')
    direction_metrics = {'angle_loss', 'direction_loss', 'angle_accuracy',
                         'direction_mae_degrees', 'sector_mae_degrees'}
    spatial_metrics = {'spatial_action_loss', 'neighbor_action_accuracy'}
    return {**{name: value / (totals['direction_samples'] if name in direction_metrics
                              else totals['moving_samples'] if name in spatial_metrics
                              else seen)
               for name, value in totals.items()
               if name not in ('direction_samples', 'moving_samples')},
            'samples': seen, 'direction_samples': int(totals['direction_samples']),
            'moving_samples': int(totals['moving_samples']),
            'seconds': time.monotonic() - started}


def train(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    initial = torch.load(args.foundation, map_location='cpu', weights_only=False)
    network_config = dict(initial.get('config', {}).get('network', {}))
    network_config['temporal_audio'] = True
    model = Network(action_count=initial['signature']['actions'], **network_config)
    incompatible = model.load_state_dict(initial['model'], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_missing = {name for name in model.state_dict()
                        if name.startswith('audio_encoder.temporal_') or
                        name.startswith('audio_encoder.direction_head.')}
    if missing != expected_missing or unexpected:
        raise RuntimeError(f'Unexpected checkpoint adaptation: missing={sorted(missing)}, '
                           f'unexpected={sorted(unexpected)}')
    # Copy the learned filters while changing only their temporal stride.
    for target, source in zip((model.audio_encoder.temporal_cnn[0],
                               model.audio_encoder.temporal_cnn[2],
                               model.audio_encoder.temporal_cnn[4]),
                              (model.audio_encoder.cnn[0],
                               model.audio_encoder.cnn[2],
                               model.audio_encoder.cnn[4])):
        target.load_state_dict(source.state_dict())
    model.to(device)

    cache, manifest = build_composite_cache(output, args.train_cache.resolve(),
                                            args.unheard_cache.resolve())
    data = loaders(cache, args.batch_size, args.workers, args.seed)
    # Keep visual and map feature extractors fixed. Fine-tune acoustics and the
    # fusion/action layers with separate conservative rates.
    frozen_prefixes = ('visual_encoder.', 'visual_transformer_encoder.',
                       'geometric_encoder.', 'acoustic_encoder.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(frozen_prefixes))
    audio = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and n.startswith('audio_encoder.')]
    fusion = [(n, p) for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith('audio_encoder.')]
    groups = []
    for named, lr in ((audio, args.audio_learning_rate), (fusion, args.fusion_learning_rate)):
        decay = [p for _, p in named if p.ndim >= 2]
        no_decay = [p for _, p in named if p.ndim < 2]
        if decay:
            groups.append({'params': decay, 'lr': lr, 'weight_decay': args.weight_decay})
        if no_decay:
            groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.})
    optimizer = torch.optim.AdamW(groups)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    signature = deepcopy(initial['signature'])
    signature['architecture'] = 'avwan_hybrid_temporal_audio_spatial_v1'
    config = {
        'network': network_config,
        'stage': 'temporal_audio_circular_direction_spatial_action_finetune',
        'initial_checkpoint': str(args.foundation.resolve()),
        'initial_checkpoint_sha256': digest(args.foundation),
        'cache_manifest': str((cache / 'manifest.json').resolve()),
        'validation_splits': ['val_heard', 'val_unheard'],
        'training': {
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'audio_learning_rate': args.audio_learning_rate,
            'fusion_learning_rate': args.fusion_learning_rate,
            'weight_decay': args.weight_decay,
            'loss': 'action CE + 0.2 legal spatial soft CE + 0.15 sector CE + 0.15 circular cosine',
            'selection': 'minimum worst validation action CE; report direction/local-action metrics',
            'weight_decay_scope': 'matrix weights only; biases and one-dimensional norm parameters excluded',
        },
    }
    write_json(output / 'config.json', config)
    train_dir = output / 'training'
    train_dir.mkdir()
    write_json(train_dir / 'initialization.json', {
        'source_checkpoint_sha256': digest(args.foundation),
        'new_parameters': sorted(missing),
        'copied_temporal_convolutions': True,
        'frozen_modules': [x[:-1] for x in frozen_prefixes],
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    })
    initial_val = {split: metric_sums(model, data[split], False, 0, device)
                   for split in ('val_heard', 'val_unheard')}
    write_json(train_dir / 'initial_validation.json', initial_val)
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        row = {'epoch': epoch,
               'train': metric_sums(model, data['train'], True, epoch, device,
                                    optimizer, scaler, args.max_train_steps)}
        for split in ('val_heard', 'val_unheard'):
            row[split] = metric_sums(model, data[split], False, epoch, device)
        row['worst_validation_action_loss'] = max(
            row['val_heard']['action_loss'], row['val_unheard']['action_loss'])
        row['worst_validation_direction_mae_degrees'] = max(
            row['val_heard']['direction_mae_degrees'],
            row['val_unheard']['direction_mae_degrees'])
        with (train_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        payload = dict(model=model.state_dict(), signature=signature, config=config,
                       epoch=epoch, validation=row,
                       selection='minimum worse heard/unheard validation action cross entropy')
        save_checkpoint(train_dir / 'latest.pt', payload)
        if row['worst_validation_action_loss'] < best:
            best = row['worst_validation_action_loss']
            save_checkpoint(train_dir / 'best.pt', payload)
            write_json(train_dir / 'best.json', {
                'epoch': epoch, 'worst_validation_action_loss': best,
                'worst_validation_direction_mae_degrees': row['worst_validation_direction_mae_degrees'],
                'checkpoint_sha256': digest(train_dir / 'best.pt'),
            })
        print(json.dumps(row), flush=True)
    write_json(output / 'status.json', {'status': 'trained', 'best_validation_action_loss': best,
                                        'checkpoint': str(train_dir / 'best.pt'),
                                        'checkpoint_sha256': digest(train_dir / 'best.pt')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation', type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument('--train-cache', type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument('--unheard-cache', type=Path, default=DEFAULT_UNHEARD_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--audio-learning-rate', type=float, default=1e-5)
    parser.add_argument('--fusion-learning-rate', type=float, default=5e-6)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--max-train-steps', type=int, default=0,
                        help='Diagnostic cap; 0 consumes the complete training split.')
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    for path in (args.foundation, args.train_cache / 'manifest.json',
                 args.unheard_cache / 'manifest.json'):
        if not path.exists():
            parser.error(f'Missing input: {path}')
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or
            min(args.audio_learning_rate, args.fusion_learning_rate, args.weight_decay) < 0):
        parser.error('Invalid training setting')
    train(args)


if __name__ == '__main__':
    main()
