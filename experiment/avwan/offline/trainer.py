import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data_pipeline.prepare import write_json
from .dataset import FeatureEpisodes, collate_episodes
from .sac_cql import RecurrentSACCQL


def save_checkpoint(path, state):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, manifest, device, resume=None):
    cfg = config['training']
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    output = Path(cfg['output'])
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Run directory not empty: {output}; use --resume or a new output')
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(architecture='independent_lstm_discrete_sac_cql_v1', features=manifest['identity'],
                     feature_dim=manifest['feature_dim'], action_count=manifest['action_count'])
    model = RecurrentSACCQL(manifest['feature_dim'], manifest['action_count'], cfg).to(device)
    if cfg.get('foundation_prior', False) and not resume:
        from .features import load_foundation, sha256
        if sha256(Path(config['features']['checkpoint'])) != manifest['identity']['checkpoint_sha256']:
            raise ValueError('Foundation prior must match cached embeddings')
        foundation = load_foundation(config['features']['checkpoint'], device)
        model.actor.prior.load_state_dict(foundation.final.action_head.state_dict())
        del foundation
    optimizers = model.make_optimizers()
    start_epoch, global_step = 0, 0
    checkpoint_index = {}
    index_path = output / 'top_checkpoints.json'
    if index_path.exists():
        for item in json.loads(index_path.read_text()):
            filename = item.get('file', '')
            if (filename.startswith('epoch_') and filename.endswith('.pt')
                    and (output / filename).is_file()):
                checkpoint_index[int(item['epoch'])] = item
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['signature'] != signature or state['config']['seed'] != config['seed']:
            raise ValueError('Resume model/feature cache/seed mismatch')
        for key in cfg:
            if key not in ('output', 'epochs', 'log_every', 'tensorboard') and cfg[key] != state['config']['training'][key]:
                raise ValueError(f'Resume hyperparameter changed: {key}')
        model.load_state_dict(state['model'])
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(state['optimizers'][key])
        start_epoch, global_step = state['epoch'], state['global_step']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda' and state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    if start_epoch >= cfg['epochs']:
        write_json(output / 'completed.json', dict(epochs=start_epoch, global_step=global_step,
                   checkpoint=str(Path(resume).resolve())))
        print(f'Checkpoint already completed {start_epoch} epochs', flush=True)
        return
    # Extending a finished run must not leave its old completion marker visible.
    (output / 'completed.json').unlink(missing_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'dataset.json', manifest)
    dataset = FeatureEpisodes(config['features']['cache'], manifest, 'train')
    print(f'SAC-CQL on {device}; full episode LSTMs; '
          f'episodes={len(dataset)}; transitions={manifest["transitions"]}', flush=True)
    writer = None
    if cfg['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = Path(config.get('tensorboard_dir', output / 'tensorboard'))
        writer = SummaryWriter(str(tensorboard_dir))

    def run_epoch(epoch):
        nonlocal global_step
        model.train()
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                            collate_fn=collate_episodes, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed'] + epoch))
        totals, seen, started = {}, 0, time.monotonic()
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            metrics = model.update(batch, optimizers)
            global_step += 1
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError('Nonfinite training metrics')
            count = int(batch['valid'].sum())
            seen += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value * count
            if step % cfg['log_every'] == 0:
                print(f'train epoch={epoch+1} batch={step}/{len(loader)} '
                      f'td={totals["td_loss"]/seen:.4f} cql={totals["cql_loss"]/seen:.4f} '
                      f'alpha={metrics["alpha"]:.4f}', flush=True)
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(f'train_step/{key}', value, global_step)
            with (output / 'steps.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(epoch=epoch + 1, global_step=global_step, **metrics)) + '\n')
        return dict(**{key: value / seen for key, value in totals.items()}, transitions=seen,
                    seconds=time.monotonic() - started)

    try:
        for epoch in range(start_epoch, cfg['epochs']):
            train_metrics = run_epoch(epoch)
            row = dict(epoch=epoch + 1, global_step=global_step, train=train_metrics)
            state = dict(epoch=epoch + 1, global_step=global_step,
                         signature=signature, config=config, model=model.state_dict(),
                         optimizers={key: value.state_dict() for key, value in optimizers.items()},
                         rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                  cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            epoch_name = f'epoch_{epoch + 1:03d}.pt'
            epoch_state = {key: value for key, value in state.items()
                           if key not in ('optimizers', 'rng')}
            epoch_state['model'] = {key: value.detach().cpu().clone()
                                    for key, value in state['model'].items()}
            save_checkpoint(output / epoch_name, epoch_state)
            save_checkpoint(output / 'latest.pt', state)
            checkpoint_index[epoch + 1] = dict(file=epoch_name, epoch=epoch + 1,
                                               criterion='epoch_snapshot')
            write_json(index_path, [checkpoint_index[key] for key in sorted(checkpoint_index)])
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if writer:
                for key, value in train_metrics.items():
                    writer.add_scalar(f'train/{key}', value, epoch + 1)
                writer.flush()
            print(json.dumps(row), flush=True)
    finally:
        if writer:
            writer.close()
    write_json(output / 'completed.json', dict(epochs=cfg['epochs'], global_step=global_step,
               checkpoint=str(output / 'latest.pt')))
    print(f'Offline RL training complete: {output}', flush=True)
