"""Recurrent BC on expert labels along real learner/expert training trajectories.

The stored transition cache retains actual behavior actions and rewards. This
trainer only optimizes actor cross-entropy, never TD losses on relabeled actions.
"""
import gzip
import argparse
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .unheard_study import OUT
from .run_gamma_study import write, sha, now
from .offline.features import extract, load_foundation
from .offline.dataset import FeatureEpisodes, collate_episodes
from .offline.sac_cql import RecurrentSACCQL, masked_policy
from .offline.trainer import save_checkpoint


def relabel_for_bc(episode, labels):
    if labels.shape != episode['action'].shape or labels.dtype != torch.long:
        raise ValueError('Expert labels do not match the trajectory')
    if not episode['action_mask'][:-1].gather(-1,labels[:,None]).all():
        raise ValueError('Expert label selects an invalid action')
    return dict(episode, action=labels.clone())


def actor_bc_loss(model, batch):
    logits, _ = model.actor(batch['features'][:, :-1])
    probs, lp = masked_policy(logits, batch['action_mask'][:, :-1])
    loss = (-lp.gather(-1, batch['action'][...,None]).squeeze(-1))[batch['valid']].mean()
    return loss, probs


def train(name='exp009-corrective-recurrent-bc', foundation=None, cache=None):
    names = ['exp007-foundation-corrective-trajectories', 'exp008-foundation-spatial-detail']
    if foundation is None:
        selected = min(names, key=lambda n: json.loads((OUT/n/'training/best.json').read_text())['val_unheard_nll'])
        foundation = OUT/selected/'training/best.pt'
        criterion = 'minimum unseen-sound validation action NLL among corrective foundations'
    else:
        foundation = Path(foundation).resolve()
        selected = foundation.parent.parent.name
        criterion = 'explicit foundation selected by independent unheard validation; see study extension record'
    folder = OUT/name;folder.mkdir(exist_ok=True)
    write(folder/'foundation_selection.json', dict(experiment=selected, checkpoint=str(foundation),
          sha256=sha(foundation), criterion=criterion))
    source = OUT/'corrective_memory_source'
    for origin, prefix in [(Path('/data/Getuanhui/avwan_multiple_greedy_train'), ''),
                           (OUT/'training_corrections', 'correction_')]:
        for path in (origin/'episodes').glob('*/*.pkl.gz'):
            dest = source/'episodes'/path.parent.name/(prefix+path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():dest.symlink_to(path.resolve())
    cfg = yaml.safe_load((OUT/'exp006-robust-recurrent-bc/config.yaml').read_text())
    cfg['features'].update(source=str(source), checkpoint=str(foundation), cache=str(cache or OUT/'corrective_memory_features'))
    cfg['training'].update(output=str(folder/'training'), epochs=60, learning_rate=1e-4,
                           weight_decay=1e-4, actor_rl_weight=0., bc_weight=1., foundation_prior=True)
    cfg['objective'] = 'actor-only recurrent cross-entropy; expert_action for correction states; no critic or TD optimization'
    cfg['correction_sampling_weight'] = 2
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    import shutil
    if not (folder/'source_snapshot').exists():
        shutil.copytree(Path(__file__).parent, folder/'source_snapshot',
                        ignore=shutil.ignore_patterns('__pycache__', 'data_cache', 'runs'))
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    device = torch.device('cuda:0')
    write(folder/'status.json', dict(status='extracting', started_at=now()))
    manifest = extract(cfg, device)
    datasets = {s: FeatureEpisodes(cfg['features']['cache'], manifest, s) for s in ['train', 'val']}
    extra = []
    for index, entry in enumerate(manifest['splits']['train']):
        if Path(entry['source']).name.startswith('correction_'):
            with gzip.open(source/entry['source'], 'rb') as f:raw = pickle.load(f)
            episode = datasets['train'].episodes[index]
            labels = torch.tensor([r['expert_action'] for r in raw['transitions']], dtype=torch.long)
            episode = relabel_for_bc(episode, labels)
            datasets['train'].episodes[index] = episode
            extra.append(episode)
    datasets['train'].episodes.extend(extra)
    signature = dict(architecture='independent_lstm_discrete_sac_cql_v1', features=manifest['identity'],
                     feature_dim=manifest['feature_dim'], action_count=manifest['action_count'])
    model = RecurrentSACCQL(manifest['feature_dim'], manifest['action_count'], cfg['training']).to(device)
    encoder = load_foundation(foundation, device)
    model.actor.prior.load_state_dict(encoder.final.action_head.state_dict());del encoder
    for module in [model.q1, model.q2, model.target_q1, model.target_q2]:module.requires_grad_(False)
    model.log_alpha.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.actor.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    output = folder/'training'
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(exist_ok=True)
    write(output/'config.json',cfg)
    write(output/'dataset.json',dict(manifest=manifest, correction_episodes=len(extra),
                                    correction_weight=2, labels='expert_action only in actor supervised objective'))
    write(folder/'status.json',dict(status='training',started_at=now()))
    def run(split,epoch):
        update = split=='train';model.train(update)
        loader = DataLoader(datasets[split],batch_size=32,shuffle=update,collate_fn=collate_episodes,
                            generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        count=0;loss_total=0.;correct=0;start=time.monotonic()
        for batch in loader:
            b={k:v.to(device) for k,v in batch.items()}
            with torch.set_grad_enabled(update):
                loss,probs=actor_bc_loss(model,b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite recurrent BC loss')
            if update:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            n=int(b['valid'].sum());count+=n;loss_total+=float(loss.detach())*n
            correct+=int(((probs.argmax(-1)==b['action'])&b['valid']).sum())
        return dict(policy_nll=loss_total/count,action_accuracy=correct/count,transitions=count,seconds=time.monotonic()-start)
    initial=run('val',0);best=initial['policy_nll']
    write(output/'initial_validation.json',initial)
    def save(epoch,val,path):
        save_checkpoint(path,dict(epoch=epoch,model=model.state_dict(),signature=signature,config=cfg,
                                  optimizer=optimizer.state_dict(),validation=val,
                                  selection='minimum independent unheard validation policy NLL, including unchanged prior at epoch zero'))
    save(0,initial,output/'best_policy.pt')
    write(output/'best.json',dict(epoch=0,val_policy_nll=best))
    for epoch in range(1,61):
        row=dict(epoch=epoch,train=run('train',epoch),val=run('val',epoch))
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        save(epoch,row['val'],output/'latest.pt')
        if row['val']['policy_nll'] < best:
            best=row['val']['policy_nll'];save(epoch,row['val'],output/'best_policy.pt')
            write(output/'best.json',dict(epoch=epoch,val_policy_nll=best))
    write(output/'completed.json',dict(epochs=60,best_validation_nll=best))
    write(folder/'status.json',dict(status='trained',completed_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='exp009-corrective-recurrent-bc')
    parser.add_argument('--foundation')
    parser.add_argument('--cache')
    args = parser.parse_args()
    train(args.name, args.foundation, args.cache)
