import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..data_pipeline.dataset import make_loader
from ..data_pipeline.prepare import write_json
from ..model.hybrid import Network
from .objectives import objective


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, manifest, resume=None):
    cfg = config['training']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if cfg['device'] == 'auto' else cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if device.type == 'cuda':
        device = torch.device('cuda', device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    output = Path(cfg['output'])
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Run directory is not empty: {output}; use --resume or a new --output')
    output.mkdir(parents=True, exist_ok=True)
    model = Network(action_count=manifest['action_count']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['amp'] and device.type == 'cuda')
    start_epoch, global_step, best = 0, 0, float('inf')
    signature = dict(identity=manifest['identity'], schema=manifest['observation_schema'],
                     actions=manifest['action_count'], architecture='avwan_hybrid_spatial_v3')
    if resume:
        # Checkpoints, like the raw pickle source, must come from a trusted run.
        state = torch.load(resume, map_location='cpu')
        if state['signature'] != signature:
            raise ValueError('Resume checkpoint model/algorithm/dataset does not match this run')
        for key in ('batch_size', 'learning_rate', 'weight_decay', 'action_loss_weight', 'angle_loss_weight', 'amp', 'max_batches'):
            if state['config']['training'][key] != cfg[key]:
                raise ValueError(f'Resume setting changed: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, global_step, best = state['epoch'], state['global_step'], state['best_loss']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda' and state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    if start_epoch >= cfg['epochs']:
        print(f'Checkpoint already completed {start_epoch} epochs', flush=True)
        return
    write_json(output / 'config.json', config)
    write_json(output / 'dataset.json', manifest)
    train_loader, val_loader = (make_loader(config, split) for split in ('train', 'val'))
    print(f'Training AVWAN hybrid.Network on {device}, samples={manifest["samples"]}, '
          f'parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)

    def run_epoch(loader, update, epoch):
        nonlocal global_step
        model.train(update)
        loader.dataset.set_epoch(epoch)
        totals, seen, data_seconds = {}, 0, 0.
        started = previous = time.monotonic()
        for step, batch in enumerate(loader, 1):
            data_seconds += time.monotonic() - previous
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.set_grad_enabled(update), torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, metrics = objective(model, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, step={step}')
            if update:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'], error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
            count = len(batch['action'])
            seen += count
            metrics['loss'] = loss
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * count
            if step % cfg['log_every'] == 0:
                print(f'{"train" if update else "val"} epoch={epoch+1} batch={step}/{len(loader)} '
                      f'loss={totals["loss"]/seen:.4f} samples/s={seen/(time.monotonic()-started):.1f}', flush=True)
            previous = time.monotonic()
            if cfg['max_batches'] and step >= cfg['max_batches']:
                break
        if not seen:
            raise ValueError('No training/validation samples')
        elapsed = time.monotonic() - started
        return dict(**{key: value / seen for key, value in totals.items()}, samples=seen,
                    seconds=elapsed, samples_per_second=seen/elapsed, data_wait_seconds=data_seconds)

    for epoch in range(start_epoch, cfg['epochs']):
        train_metrics = run_epoch(train_loader, True, epoch)
        val_metrics = run_epoch(val_loader, False, epoch)
        improved = val_metrics['loss'] < best
        best = min(best, val_metrics['loss'])
        row = dict(epoch=epoch + 1, global_step=global_step, train=train_metrics, val=val_metrics)
        print(json.dumps(row), flush=True)
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        state = dict(epoch=epoch + 1, global_step=global_step, best_loss=best, signature=signature,
                     config=config, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                              cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
        save_checkpoint(output / 'latest.pt', state)
        if improved:
            save_checkpoint(output / 'best.pt', state)
            save_checkpoint(output / 'foundation_weights.pt', model.state_dict())
    print(f'Training complete. Checkpoints: {output}', flush=True)
