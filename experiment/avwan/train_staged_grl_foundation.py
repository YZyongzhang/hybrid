"""Train audio with a heard/unheard GRL head, then freeze audio for actions."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import resource
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avwan.data_pipeline.dataset import make_loader
from experiment.avwan.data_pipeline.prepare import INPUTS, prepare as prepare_episodes, write_json
from experiment.avwan.model.hybrid import Network
from experiment.avwan.model.maps import unpack_geometry
from experiment.avwan.training.trainer import save_checkpoint


DEFAULT_TRAIN_SOURCE = ROOT / 'result/10-11/total/2026-10-11-large-train-snapshot'
DEFAULT_HEARD_SOURCE = Path('/data/Getuanhui/avwan_multiple_greedy_val_heard')
NETWORK_CONFIG = {
    'audio_normalization': 'binaural_ratio',
    'map_normalization': 'relative',
    'spatial_detail': True,
}


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _prepare_source(source, destination, seed, val_source=None):
    cfg = {
        'seed': seed,
        'data': {
            'source': str(Path(source).resolve()),
            'val_source': str(Path(val_source).resolve()) if val_source else '',
            'cache': str(destination.resolve()),
            'inputs': list(INPUTS),
            'shard_size': 512,
            'convert_workers': 4,
            'val_fraction': .1,
            'split_by': 'scene',
            'max_episodes': 0,
        },
    }
    return prepare_episodes(cfg)


def _sound_for_episode(path):
    with gzip.open(path, 'rb') as handle:
        episode = pickle.load(handle)
    metadata = episode.get('metadata', {})
    sound = metadata.get('sound')
    transitions = metadata.get('transitions', len(episode.get('transitions', ())))
    if not sound or transitions != len(episode.get('transitions', ())):
        raise ValueError(f'Missing or inconsistent sound metadata: {path}')
    return sound, transitions


def prepare_data(output, seed, train_source, heard_source, unheard_source=None):
    """Rebuild episode tensors, sound labels, and validation manifests from raw episodes."""
    prepared = output / 'prepared_data'
    if prepared.exists():
        shutil.rmtree(prepared)
    prepared.mkdir(parents=True)
    train_data = prepared / 'train_and_heard'
    base = _prepare_source(train_source, train_data, seed, heard_source)
    base_manifest = json.loads((train_data / 'manifest.json').read_text())

    source_rows = json.loads((train_data / 'sources.json').read_text())['train']
    sound_by_path = {}
    for relative, _, _ in source_rows:
        sound_by_path[relative] = _sound_for_episode(Path(train_source).resolve() / relative)
    sounds = sorted({sound for sound, _ in sound_by_path.values()})
    class_to_id = {name: index for index, name in enumerate(sounds)}
    labels_dir = prepared / 'sound_labels'
    labels_dir.mkdir()
    label_shards, class_samples = [], {name: 0 for name in sounds}
    position = 0
    shuffled_source_rows = list(source_rows)
    random.Random(seed).shuffle(shuffled_source_rows)
    all_labels = np.concatenate([
        np.full(count, class_to_id[sound], dtype=np.int16)
        for sound, count in (sound_by_path[row[0]] for row in shuffled_source_rows)
    ])
    if len(all_labels) != base_manifest['samples']['train']:
        raise ValueError('Sound-label count differs from freshly prepared train episodes')
    for index, shard in enumerate(base_manifest['splits']['train']):
        count = shard['samples']
        selected = all_labels[position:position + count]
        source_dir = train_data / shard['path']
        previous = np.load(source_dir / 'previous_index.npy', mmap_mode='r')
        changes = np.flatnonzero(selected[1:] != selected[:-1]) + 1
        if np.any(previous[changes] < count):
            raise ValueError(f'Sound labels do not align with episode boundaries: {shard["path"]}')
        previous._mmap.close()
        label_path = source_dir.parent / f'{source_dir.name}.npy'
        np.save(label_path, selected, allow_pickle=False)
        label_shards.append(dict(path=str(label_path.resolve()), samples=count,
                                 sha256=sha256(label_path)))
        position += count
    for sound, count in sound_by_path.values():
        class_samples[sound] += count
    labels = {
        'cache': str(train_data.resolve()),
        'cache_fingerprint': base_manifest['identity']['fingerprint'],
        'episodes': len(sound_by_path), 'samples': len(all_labels),
        'classes': sounds, 'class_samples': [class_samples[name] for name in sounds],
        'shards': label_shards,
    }
    write_json(labels_dir / 'manifest.json', labels)

    rng = random.Random(seed)
    order = list(sounds)
    rng.shuffle(order)
    pseudo_unheard = sorted(order[:len(order) // 2])
    pseudo_heard = sorted(set(sounds) - set(pseudo_unheard))

    cache = prepared / 'dataset'
    cache.mkdir()
    manifest = deepcopy(base_manifest)
    manifest['splits'] = {
        'train': [dict(row, path=str((train_data / row['path']).resolve()))
                  for row in base_manifest['splits']['train']],
        'val_heard': [dict(row, path=str((train_data / row['path']).resolve()))
                      for row in base_manifest['splits']['val']],
    }
    manifest['samples'] = {
        'train': base_manifest['samples']['train'],
        'val_heard': base_manifest['samples']['val'],
    }
    manifest['prepared_sources'] = {
        'train': str(Path(train_source).resolve()),
        'val_heard': str(Path(heard_source).resolve()),
    }
    validation_splits = ['val_heard']
    if unheard_source is not None:
        unheard_data = prepared / 'unheard_validation'
        unheard = _prepare_source(unheard_source, unheard_data, seed)
        unheard_entries = []
        # Include all independent unheard episodes; prepare() also emits a
        # secondary split that is not needed for validation-only data.
        for split in ('train', 'val'):
            unheard_entries.extend(
                dict(row, path=str((unheard_data / row['path']).resolve()))
                for row in unheard['splits'][split]
            )
        manifest['splits']['val'] = unheard_entries
        manifest['samples']['val'] = sum(row['samples'] for row in unheard_entries)
        manifest['independent_unheard_validation'] = {
            'source': str(Path(unheard_source).resolve()),
            'episodes': sum(len(unheard['splits'][split]) for split in ('train', 'val')),
            'training_use': False,
        }
        manifest['prepared_sources']['val_unheard'] = str(Path(unheard_source).resolve())
        validation_splits.insert(0, 'val')
    (cache / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    split_info = {
        'method': 'deterministic class-disjoint pseudo-unheard split of already collected training episodes',
        'seed': seed,
        'pseudo_unheard_sound_classes': pseudo_unheard,
        'pseudo_heard_sound_classes': pseudo_heard,
        'pseudo_unheard_class_count': len(pseudo_unheard),
        'pseudo_heard_class_count': len(pseudo_heard),
        'train_samples': base_manifest['samples']['train'],
        'train_episodes': labels['episodes'],
        'class_sample_counts': dict(zip(sounds, labels['class_samples'])),
    }
    write_json(output / 'pseudo_unheard_split.json', split_info)
    return cache, labels_dir, labels, split_info, validation_splits


def memory_guard(max_rss_mib, min_available_gib):
    with Path('/proc/meminfo').open() as handle:
        available_kib = next(int(line.split()[1]) for line in handle if line.startswith('MemAvailable:'))
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    available_gib = available_kib / (1024 * 1024)
    if rss_mib >= max_rss_mib or available_gib <= min_available_gib:
        raise MemoryError(f'Resource guard stopped training: RSS={rss_mib:.0f} MiB, '
                          f'available RAM={available_gib:.1f} GiB')


def loaders(cache, labels_dir, batch_size, workers, seed, validation_splits):
    config = {
        'seed': seed,
        'data': {'cache': str(cache), 'sound_labels': str(labels_dir)},
        'training': {'batch_size': batch_size, 'workers': workers,
                     'pin_memory': False, 'prefetch_factor': 1},
    }
    return {split: make_loader(config, split)
            for split in ('train', *validation_splits)}


def audio_epoch(model, classifier, loader, update, epoch, device, optimizer=None,
                scaler=None, label_to_domain=None, grl_scale=0.2,
                domain_weight=1.0, max_rss_mib=12000, min_available_gib=12.0):
    model.train(update)
    if not update:
        model.eval()
    if classifier is not None:
        classifier.train(update)
    loader.dataset.set_epoch(epoch)
    totals, seen, valid_seen = {}, 0, 0
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        spec = batch['spectrogram'].to(device, non_blocking=False)
        action = batch['action'].to(device, non_blocking=False)
        angle_label = batch['angle_label'].to(device, non_blocking=False)
        valid_angle = action != model.action_count // 2
        with torch.set_grad_enabled(update), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            token = model.audio_encoder.encoder_forward(spec).squeeze(1)
            angle_logits = model.audio_encoder.fnn(token.float())
            angle_loss = (F.cross_entropy(angle_logits[valid_angle], angle_label[valid_angle])
                          if valid_angle.any() else angle_logits.sum() * 0.)
            loss = angle_loss
            metrics = {'angle_loss': angle_loss,
                       'angle_accuracy': ((angle_logits.argmax(-1) == angle_label) & valid_angle).sum()
                                          / valid_angle.sum().clamp_min(1)}
            if update:
                labels = batch.get('sound_label')
                if labels is None or label_to_domain is None:
                    raise ValueError('GRL audio training requires aligned sound labels')
                sound_label = labels.to(device, non_blocking=False)
                domain = label_to_domain[sound_label]
                domain_logits = classifier(ReverseGradient.apply(token.float(), grl_scale))
                domain_loss = F.cross_entropy(domain_logits, domain)
                loss = loss + domain_weight * domain_loss
                metrics.update(domain_loss=domain_loss,
                               domain_accuracy=(domain_logits.argmax(-1) == domain).float().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite audio-stage loss')
        if update:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.audio_encoder.parameters() if p.requires_grad] +
                list(classifier.parameters()), 10.)
            scaler.step(optimizer)
            scaler.update()
        count = len(action)
        seen += count
        valid_seen += int(valid_angle.sum())
        for key, value in dict(loss=loss, **metrics).items():
            totals[key] = totals.get(key, 0.) + float(value.detach()) * count
        if step % 1000 == 0:
            memory_guard(max_rss_mib, min_available_gib)
    return dict(**{key: value / seen for key, value in totals.items()},
                samples=seen, direction_samples=valid_seen,
                seconds=time.monotonic() - started)


def full_batch(model, batch, device):
    moved = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
    rgb = torch.cat((moved['prev_rgb'], moved['rgb']), dim=-1).float() / 255.
    depth = torch.cat((moved['prev_depth'], moved['depth']), dim=-1).float()
    logits, angles = model.forward_joint(moved['spectrogram'], rgb, depth,
                                         unpack_geometry(moved['gm']), moved['am'],
                                         moved['action_mask'])
    return logits, angles, moved


def action_epoch(model, loader, update, epoch, device, optimizer=None, scaler=None,
                 max_rss_mib=12000, min_available_gib=12.0):
    model.train(update)
    model.audio_encoder.eval()  # Keep the complete audio branch frozen and deterministic.
    loader.dataset.set_epoch(epoch)
    totals, seen = {}, 0
    started = time.monotonic()
    for step, batch in enumerate(loader, 1):
        with torch.set_grad_enabled(update), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, _, moved = full_batch(model, batch, device)
            loss = F.cross_entropy(logits.float(), moved['action'])
            accuracy = (logits.argmax(-1) == moved['action']).float().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite action-stage loss')
        if update:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 10.)
            scaler.step(optimizer)
            scaler.update()
        count = len(moved['action'])
        seen += count
        totals['action_loss'] = totals.get('action_loss', 0.) + float(loss.detach()) * count
        totals['action_accuracy'] = totals.get('action_accuracy', 0.) + float(accuracy.detach()) * count
        if step % 1000 == 0:
            memory_guard(max_rss_mib, min_available_gib)
    return dict(**{key: value / seen for key, value in totals.items()},
                samples=seen, seconds=time.monotonic() - started)


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        status_path = output / 'status.json'
        audio_latest = output / 'foundation/audio_stage/latest.pt'
        if (not status_path.exists() or json.loads(status_path.read_text()).get('status') != 'audio_training'
                or audio_latest.exists()):
            raise FileExistsError(f'Choose an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    memory_guard(args.max_rss_mib, args.min_available_gib)

    cache, label_dir, labels, split_info, validation_splits = prepare_data(
        output, args.seed, args.train_source, args.heard_validation_source,
        args.unheard_validation_source)
    cache_manifest = json.loads((cache / 'manifest.json').read_text())
    network = dict(NETWORK_CONFIG)
    model = Network(action_count=cache_manifest['action_count'], **network)
    model.to(device)
    classifier = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
    classes = labels['classes']
    pseudo = set(split_info['pseudo_unheard_sound_classes'])
    domain_map = torch.tensor([int(name in pseudo) for name in classes], dtype=torch.long, device=device)
    data_loaders = loaders(cache, label_dir, args.batch_size, args.workers, args.seed,
                           validation_splits)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    audio_parameters = list(model.audio_encoder.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(audio_parameters, lr=args.audio_learning_rate, weight_decay=1e-4)
    audio_dir = output / 'foundation/audio_stage'
    audio_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        'identity': cache_manifest['identity'],
        'schema': cache_manifest['observation_schema'],
        'actions': cache_manifest['action_count'],
        'architecture': 'avwan_hybrid_acoustic_domain_v1',
    }
    config = {
        'initialization': 'random',
        'data_cache': str(cache), 'sound_labels': str(label_dir),
        'network': network, 'seed': args.seed, 'pseudo_unheard_split': split_info,
        'memory': {'gpu_memory_fraction': args.gpu_memory_fraction,
                   'batch_size': args.batch_size, 'workers': args.workers,
                   'max_rss_mib': args.max_rss_mib,
                   'min_available_gib': args.min_available_gib},
        'audio_stage': {'epochs': args.audio_epochs,
                        'learning_rate': args.audio_learning_rate,
                        'grl_scale': args.grl_scale,
                        'domain_loss_weight': args.domain_loss_weight,
                        'validation_splits': validation_splits,
                        'selection': 'minimum worst validation angle cross entropy'},
        'action_stage': {'epochs': args.action_epochs,
                         'learning_rate': args.action_learning_rate,
                         'validation_splits': validation_splits,
                         'selection': 'minimum worst validation action cross entropy',
                         'audio_frozen': True},
    }
    write_json(output / 'experiment_config.json', config)
    write_json(output / 'status.json', {'status': 'audio_training'})
    write_json(audio_dir / 'initial_validation.json', {
        split: audio_epoch(model, classifier, data_loaders[split], False, 0, device,
                           max_rss_mib=args.max_rss_mib,
                           min_available_gib=args.min_available_gib)
        for split in validation_splits})
    best_audio = float('inf')
    for epoch in range(1, args.audio_epochs + 1):
        row = {
            'epoch': epoch,
            'train': audio_epoch(model, classifier, data_loaders['train'], True, epoch,
                                 device, optimizer, amp_scaler, domain_map,
                                 args.grl_scale, args.domain_loss_weight,
                                 args.max_rss_mib, args.min_available_gib),
        }
        row.update({
            split: audio_epoch(model, classifier, data_loaders[split], False, epoch, device,
                               max_rss_mib=args.max_rss_mib,
                               min_available_gib=args.min_available_gib)
            for split in validation_splits
        })
        row['worst_validation_angle_loss'] = max(
            row[split]['angle_loss'] for split in validation_splits)
        with (audio_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        payload = dict(model=model.state_dict(), sound_classifier=classifier.state_dict(),
                       signature=deepcopy(signature), config=config, epoch=epoch,
                       validation=row, selection='minimum worst validation angle loss')
        save_checkpoint(audio_dir / 'latest.pt', payload)
        if row['worst_validation_angle_loss'] < best_audio:
            best_audio = row['worst_validation_angle_loss']
            save_checkpoint(audio_dir / 'best.pt', payload)
            write_json(audio_dir / 'best.json', {'epoch': epoch,
                       'worst_validation_angle_loss': best_audio,
                       'checkpoint_sha256': sha256(audio_dir / 'best.pt')})
        print(json.dumps({'stage': 'audio', **row}), flush=True)

    write_json(output / 'status.json', {'status': 'action_training'})
    audio_state = torch.load(audio_dir / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(audio_state['model'], strict=True)
    model.requires_grad_(True)
    model.audio_encoder.requires_grad_(False)
    model.audio_encoder.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    action_optimizer = torch.optim.AdamW(trainable, lr=args.action_learning_rate, weight_decay=1e-4)
    action_scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    action_dir = output / 'foundation/action_stage'
    action_dir.mkdir(parents=True)
    write_json(action_dir / 'trainable_parameters.json', {
        'trainable': sum(p.numel() for p in trainable),
        'frozen_audio': sum(p.numel() for p in model.audio_encoder.parameters()),
        'trainable_names': [name for name, parameter in model.named_parameters()
                            if parameter.requires_grad],
    })
    write_json(action_dir / 'initial_validation.json', {
        split: action_epoch(model, data_loaders[split], False, 0, device,
                            max_rss_mib=args.max_rss_mib,
                            min_available_gib=args.min_available_gib)
        for split in validation_splits})
    best_action = float('inf')
    for epoch in range(1, args.action_epochs + 1):
        row = {
            'epoch': epoch,
            'train': action_epoch(model, data_loaders['train'], True, epoch, device,
                                  action_optimizer, action_scaler, args.max_rss_mib,
                                  args.min_available_gib),
        }
        row.update({
            split: action_epoch(model, data_loaders[split], False, epoch, device,
                                max_rss_mib=args.max_rss_mib,
                                min_available_gib=args.min_available_gib)
            for split in validation_splits
        })
        row['worst_validation_action_loss'] = max(
            row[split]['action_loss'] for split in validation_splits)
        with (action_dir / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        payload = dict(model=model.state_dict(), signature=deepcopy(signature),
                       config=dict(network=network, stage='full_action_after_grl_audio',
                                   audio_checkpoint=str(audio_dir / 'best.pt'),
                                   audio_checkpoint_sha256=sha256(audio_dir / 'best.pt'),
                                   training=config['action_stage']), epoch=epoch,
                       validation=row,
                       selection='minimum worst validation action cross entropy')
        save_checkpoint(action_dir / 'latest.pt', payload)
        if row['worst_validation_action_loss'] < best_action:
            best_action = row['worst_validation_action_loss']
            save_checkpoint(action_dir / 'best.pt', payload)
            write_json(action_dir / 'best.json', {'epoch': epoch,
                       'worst_validation_action_loss': best_action,
                       'checkpoint_sha256': sha256(action_dir / 'best.pt')})
        print(json.dumps({'stage': 'action', **row}), flush=True)

    audio_frozen_names = []
    for name, parameter in model.named_parameters():
        if name.startswith('audio_encoder.'):
            if parameter.requires_grad:
                raise AssertionError(f'Audio parameter was not frozen: {name}')
            if not torch.equal(parameter.detach().cpu(), audio_state['model'][name]):
                raise AssertionError(f'Frozen audio parameter changed: {name}')
            audio_frozen_names.append(name)
    write_json(action_dir / 'frozen_audio_audit.json', {
        'passed': True, 'checked_parameters': audio_frozen_names,
        'audio_checkpoint': str(audio_dir / 'best.pt'),
        'audio_checkpoint_sha256': sha256(audio_dir / 'best.pt'),
    })
    write_json(output / 'status.json', {
        'status': 'foundation_trained',
        'best_audio_validation_angle_loss': best_audio,
        'best_action_validation_action_loss': best_action,
        'foundation_checkpoint': str(action_dir / 'best.pt'),
        'foundation_checkpoint_sha256': sha256(action_dir / 'best.pt'),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-source', type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument('--heard-validation-source', type=Path, default=DEFAULT_HEARD_SOURCE)
    parser.add_argument('--unheard-validation-source', type=Path)
    parser.add_argument('--audio-epochs', type=int, default=5)
    parser.add_argument('--action-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--audio-learning-rate', type=float, default=1e-4)
    parser.add_argument('--action-learning-rate', type=float, default=3e-5)
    parser.add_argument('--grl-scale', type=float, default=.2)
    parser.add_argument('--domain-loss-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=.22)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--max-rss-mib', type=int, default=12000)
    parser.add_argument('--min-available-gib', type=float, default=12.)
    args = parser.parse_args()
    for source in (args.train_source, args.heard_validation_source,
                   args.unheard_validation_source):
        if source is None:
            continue
        if not source.is_dir():
            parser.error(f'Raw episode source directory does not exist: {source}')
    if (min(args.audio_epochs, args.action_epochs, args.batch_size, args.cpu_threads) < 1 or
            args.workers < 0 or args.audio_learning_rate <= 0 or args.action_learning_rate <= 0 or
            args.grl_scale < 0 or args.domain_loss_weight < 0 or
            not 0 < args.gpu_memory_fraction <= 1 or args.max_rss_mib < 1 or
            args.min_available_gib < 0):
        parser.error('Invalid training or memory-safety setting')
    run(args)


if __name__ == '__main__':
    main()
