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
    optimizers = model.make_optimizers()
    start_epoch, global_step, best, best_nll = 0, 0, float('inf'), float('inf')
    candidates = {}
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
        start_epoch, global_step, best = state['epoch'], state['global_step'], state['best_val_critic_loss']
        best_nll = state.get('best_val_policy_nll', float('inf'))
        candidates = state.get('candidates', {})
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if device.type == 'cuda' and state['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
    if start_epoch >= cfg['epochs']:
        write_json(output / 'completed.json', dict(epochs=start_epoch, global_step=global_step,
                   best_val_critic_loss=best, checkpoint=str(Path(resume).resolve())))
        print(f'Checkpoint already completed {start_epoch} epochs', flush=True)
        return
    # Extending a finished run must not leave its old completion marker visible.
    (output / 'completed.json').unlink(missing_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'dataset.json', manifest)
    datasets = {split: FeatureEpisodes(config['features']['cache'], manifest, split) for split in ('train', 'val')}
    print(f'SAC-CQL on {device}; full episode LSTMs; episodes='
          f'{ {k: len(v) for k, v in datasets.items()} }; transitions={manifest["transitions"]}', flush=True)
    writer = None
    if cfg['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / 'tensorboard'))

    def run_epoch(split, epoch):
        nonlocal global_step
        update = split == 'train'
        model.train(update)
        loader = DataLoader(datasets[split], batch_size=cfg['batch_size'], shuffle=update,
                            collate_fn=collate_episodes, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed'] + epoch))
        totals, seen, started = {}, 0, time.monotonic()
        for step, batch in enumerate(loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            if update:
                metrics = model.update(batch, optimizers)
                global_step += 1
            else:
                with torch.no_grad():
                    metrics = {key: float(value) for key, value in model.losses(batch).items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f'Nonfinite {split} metrics')
            count = int(batch['valid'].sum())
            seen += count
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value * count
            if update and step % cfg['log_every'] == 0:
                print(f'train epoch={epoch+1} batch={step}/{len(loader)} '
                      f'td={totals["td_loss"]/seen:.4f} cql={totals["cql_loss"]/seen:.4f} '
                      f'alpha={metrics["alpha"]:.4f}', flush=True)
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(f'train_step/{key}', value, global_step)
            if update:
                with (output / 'steps.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(epoch=epoch + 1, global_step=global_step, **metrics)) + '\n')
        return dict(**{key: value / seen for key, value in totals.items()}, transitions=seen,
                    seconds=time.monotonic() - started)

    try:
        for epoch in range(start_epoch, cfg['epochs']):
            train_metrics = run_epoch('train', epoch)
            val_metrics = run_epoch('val', epoch)
            improved = val_metrics['critic_loss'] < best
            improved_nll = val_metrics['policy_nll'] < best_nll
            best = min(best, val_metrics['critic_loss'])
            best_nll = min(best_nll, val_metrics['policy_nll'])
            for name, criterion, changed in [('best.pt', 'val_critic_loss', improved),
                                             ('best_policy.pt', 'val_policy_nll', improved_nll),
                                             ('latest.pt', 'final_epoch', True)]:
                if changed:
                    candidates[name] = dict(file=name, epoch=epoch + 1, criterion=criterion,
                                            val_loss=val_metrics['critic_loss'],
                                            val_policy_nll=val_metrics['policy_nll'])
            row = dict(epoch=epoch + 1, global_step=global_step, train=train_metrics, val=val_metrics)
            state = dict(epoch=epoch + 1, global_step=global_step, best_val_critic_loss=best,
                         best_val_policy_nll=best_nll, candidates=candidates.copy(),
                         signature=signature, config=config, model=model.state_dict(),
                         optimizers={key: value.state_dict() for key, value in optimizers.items()},
                         rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                  cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            save_checkpoint(output / 'latest.pt', state)
            if improved:
                save_checkpoint(output / 'best.pt', state)
            if improved_nll:
                save_checkpoint(output / 'best_policy.pt', state)
            write_json(output / 'top_checkpoints.json', list(candidates.values()))
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if writer:
                for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
                    for key, value in metrics.items():
                        writer.add_scalar(f'{split}/{key}', value, epoch + 1)
                writer.flush()
            print(json.dumps(row), flush=True)
    finally:
        if writer:
            writer.close()
    write_json(output / 'completed.json', dict(epochs=cfg['epochs'], global_step=global_step,
               best_val_critic_loss=best, checkpoint=str(output / 'latest.pt')))
    print(f'Offline RL training complete: {output}', flush=True)
