"""Fine-tune pretrained AVWAN with shared-ear acoustic domain augmentation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data_pipeline.dataset import make_loader
from .data_pipeline.prepare import read_episode, INPUTS, write_json
from .model.hybrid import Network
from .training.objectives import objective
from .training.trainer import save_checkpoint
from .unheard_study import OUT, ROOT
from .run_gamma_study import sha, now

BASE_CACHE = ROOT/'result/9-29-multiple-foundation/data_cache'
FOUNDATION = ROOT/'result/9-29-multiple-foundation/best_foundation.pt'


def validation_cache():
    """Reference immutable training shards; only write independent validation."""
    cache = OUT/'domain_cache'
    if (cache/'manifest.json').exists():
        return cache
    paths = sorted((OUT/'validation_demonstrations/episodes').glob('*/*.pkl.gz'))
    if len(paths) != 100:
        raise ValueError(f'Validation collection unfinished: {len(paths)}/100')
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE_CACHE/'manifest.json').read_text())
    for split in manifest['splits'].values():
        for shard in split:
            shard['path'] = str(BASE_CACHE/shard['path'])
    manifest['splits']['val_heard'] = manifest['splits'].pop('val')
    shards = []
    for index, path in enumerate(paths):
        rows, _, _ = read_episode(path, INPUTS)
        folder = cache/'val'/f'{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        count = len(rows)
        for key in ['rgb', 'depth']:
            np.save(folder/(key+'.npy'), np.stack([r[key] for r in rows]+[np.zeros_like(rows[0][key])]))
        previous = np.arange(count, dtype=np.int64)-1
        previous[0] = count
        np.save(folder/'previous_index.npy', previous)
        for key in ['spectrogram', 'gm', 'am', 'action', 'action_mask', 'angle_label']:
            np.save(folder/(key+'.npy'), np.stack([r[key] for r in rows]))
        shards.append(dict(path=str(folder), samples=count))
    manifest['splits']['val'] = shards
    manifest['samples']['val_heard'] = manifest['samples']['val']
    manifest['samples']['val'] = sum(s['samples'] for s in shards)
    manifest['independent_unheard_validation'] = dict(source=str(OUT/'validation_demonstrations'),
                                                    episodes=100, training_use=False)
    write_json(cache/'manifest.json', manifest)
    return cache


def augment_acoustics(batch, strength=1.):
    """Shared gain and source-spectrum coloration; never normalize ears apart."""
    result = dict(batch)
    magnitude = torch.expm1(batch['spectrogram'].float().clamp(0, 20))
    n, frequencies = magnitude.shape[:2]
    device = magnitude.device
    # Low-dimensional smooth EQ plus random training-sample spectral envelope.
    knots = torch.randn(n, 1, 8, device=device) * (.8 * strength)
    log_eq = F.interpolate(knots, size=frequencies, mode='linear', align_corners=True).transpose(1, 2).unsqueeze(-1)
    envelope = magnitude.mean((2, 3), keepdim=True).clamp_min(1e-6).log()
    envelope = envelope - envelope.mean(1, keepdim=True)
    other = envelope[torch.randperm(n, device=device)]
    mixed = magnitude * (log_eq + .5*strength*(other-envelope).clamp(-4, 4)).exp()
    # Hold total energy fixed during EQ so the map's gain remains coherent.
    rms = lambda x: x.square().mean((1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt()
    mixed = mixed * rms(magnitude) / rms(mixed)
    gain = torch.exp((torch.rand(n, 1, 1, 1, device=device)*2-1) * (3.45*strength))
    result['spectrogram'] = torch.log1p(mixed*gain)
    result['am'] = batch['am'].float() * gain.square()
    return result


def train(name, mode, epochs=40, learning_rate=1e-4, checkpoint=None, data_cache=None, spatial_detail=False, freeze_backbone=False):
    cache = Path(data_cache) if data_cache else validation_cache()
    initialization = Path(checkpoint) if checkpoint else FOUNDATION
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    output = folder/'training'
    if output.exists():
        raise FileExistsError(f'Already exists: {output}')
    output.mkdir()
    seed = 20261003
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    network = dict(audio_normalization={'whiten':'binaural_whiten','ratio':'binaural_ratio'}.get(mode,'standard'),
                   map_normalization='absolute' if mode == 'gain' else 'relative')
    if spatial_detail:
        network['spatial_detail'] = True
    config = dict(seed=seed, network=network, data=dict(cache=str(cache)),
                  augmentation=dict(mode=mode, strength=1., common_ear_gain_log_range=3.45),
                  initialization=dict(checkpoint=str(initialization), sha256=sha(initialization)),
                  training=dict(output=str(output), epochs=epochs, batch_size=64, workers=4,
                                prefetch_factor=2, pin_memory=True, learning_rate=learning_rate,
                                weight_decay=1e-4, action_loss_weight=1., angle_loss_weight=.5,
                                freeze_backbone=freeze_backbone,angle_nonstop_only=freeze_backbone))
    write_json(folder/'config.json', config)
    write_json(folder/'status.json', dict(status='training', started_at=now()))
    import shutil
    shutil.copytree(ROOT/'experiment/avwan', folder/'source_snapshot',
                    ignore=shutil.ignore_patterns('__pycache__', 'data_cache', 'runs'), dirs_exist_ok=True)
    state = torch.load(initialization, map_location='cpu', weights_only=False)
    model = Network(**network).to(device)
    loaded = model.load_state_dict(state['model'], strict=False)
    if loaded.unexpected_keys or any(not k.startswith('detail_encoder.') for k in loaded.missing_keys):
        raise ValueError(f'Unexpected initialization mismatch: {loaded}')
    frozen = []
    if freeze_backbone:
        model.requires_grad_(False)
        for module in [model.audio_encoder,model.detail_encoder,model.final.action_head]:
            if module is not None:module.requires_grad_(True)
        frozen = [model.visual_encoder,model.visual_transformer_encoder,model.geometric_encoder,
                  model.acoustic_encoder,model.mask_encoder,model.vaencoder,model.final.fc1]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4)
    write_json(output/'trainable_parameters.json',dict(names=[n for n,p in model.named_parameters() if p.requires_grad],
               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),total=sum(p.numel() for p in model.parameters())))
    scaler = torch.cuda.amp.GradScaler()
    loaders = {split: make_loader(config, split) for split in ['train', 'val', 'val_heard']}
    best = float('inf')
    signature = deepcopy(state['signature'])
    signature['architecture'] = 'avwan_hybrid_acoustic_domain_v1'

    def run(loader, update, epoch):
        model.train(update)
        for module in frozen:module.eval()
        loader.dataset.set_epoch(epoch)
        seen, totals = 0, {}
        start = time.monotonic()
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if update:
                batch = augment_acoustics(batch)
            with torch.set_grad_enabled(update), torch.cuda.amp.autocast():
                loss, metrics = objective(model, batch, config['training'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            if update:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                scaler.step(optimizer); scaler.update()
            n = len(batch['action']); seen += n
            for key, value in dict(loss=loss, **metrics).items():
                totals[key] = totals.get(key, 0.) + float(value.detach())*n
        return dict(**{k: v/seen for k, v in totals.items()}, samples=seen, seconds=time.monotonic()-start)

    # Record initial out-of-domain behavior before gradient updates.
    write_json(output/'initial_validation.json', {s: run(loaders[s], False, 0) for s in ['val', 'val_heard']})
    for epoch in range(1, epochs+1):
        row = dict(epoch=epoch, **{split: run(loader, split=='train', epoch) for split, loader in loaders.items()})
        with (output/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        score = row['val']['action_loss']
        payload = dict(model=model.state_dict(), signature=signature, config=config, epoch=epoch,
                       optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), validation=row,
                       selection='minimum independent unheard validation action NLL')
        save_checkpoint(output/'latest.pt', payload)
        if score < best:
            best = score
            save_checkpoint(output/'best.pt', payload)
            write_json(output/'best.json', dict(epoch=epoch, val_unheard_nll=score, sha256=sha(output/'best.pt')))
        if epoch % 10 == 0:
            save_checkpoint(output/f'epoch_{epoch:03d}.pt', payload)
    if freeze_backbone:
        checked=[]
        for name,parameter in model.named_parameters():
            if not parameter.requires_grad:
                assert torch.equal(parameter.detach().cpu(),state['model'][name]),name
                checked.append(name)
        write_json(output/'frozen_backbone_audit.json',dict(passed=True,checked=checked))
    write_json(folder/'status.json', dict(status='trained', completed_at=now(), best_validation_nll=best))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--mode', choices=['gain', 'relative', 'whiten', 'ratio'], required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--checkpoint')
    p.add_argument('--cache')
    p.add_argument('--spatial-detail', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    a = p.parse_args()
    train(a.name, a.mode, a.epochs, a.learning_rate, a.checkpoint, a.cache, a.spatial_detail, a.freeze_backbone)


if __name__ == '__main__':
    main()
