"""Compact feature episodes; full causal histories without cross-episode windows."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureEpisodes(Dataset):
    def __init__(self, cache, manifest, split):
        # The expensive sensory data is gone. Keep compact features in RAM so
        # epochs do not repeatedly open thousands of tiny files.
        self.episodes = []
        for entry in manifest['splits'][split]:
            with np.load(Path(cache) / entry['path'], allow_pickle=False) as data:
                episode = {key: torch.from_numpy(data[key].copy()) for key in
                           ('features', 'action_mask', 'action', 'reward', 'discount')}
            n = len(episode['action'])
            if (n != entry['transitions'] or n < 1
                    or episode['features'].shape != (n + 1, manifest['feature_dim'])
                    or episode['action_mask'].shape != (n + 1, manifest['action_count'])
                    or episode['action_mask'].dtype != torch.bool
                    or not episode['action_mask'].any(-1).all()
                    or not torch.isfinite(episode['features']).all()
                    or episode['action'].dtype != torch.int64
                    or (episode['action'] < 0).any()
                    or (episode['action'] >= manifest['action_count']).any()):
                raise ValueError(f'Invalid cached episode: {entry["path"]}')
            for key in ('reward', 'discount'):
                if episode[key].shape != (n,) or not torch.isfinite(episode[key]).all():
                    raise ValueError(f'Invalid {key}: {entry["path"]}')
            if (not episode['action_mask'][:-1].gather(-1, episode['action'][:, None]).all()
                    or (episode['discount'] < 0).any() or (episode['discount'] > 1).any()):
                raise ValueError(f'Invalid action/discount: {entry["path"]}')
            from .reward_shaping import resolve_source
            episode['_source_path'] = str(resolve_source(manifest, entry).resolve())
            self.episodes.append(episode)
        if not self.episodes:
            raise ValueError(f'Empty {split} feature dataset')

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        return self.episodes[index]


def collate_episodes(episodes):
    batch_size = len(episodes)
    length = max(len(e['action']) for e in episodes)
    feature_dim = episodes[0]['features'].shape[-1]
    actions = episodes[0]['action_mask'].shape[-1]
    batch = dict(features=torch.zeros(batch_size, length + 1, feature_dim),
                 action_mask=torch.ones(batch_size, length + 1, actions, dtype=torch.bool),
                 action=torch.zeros(batch_size, length, dtype=torch.long),
                 stop_target=torch.zeros(batch_size, length, dtype=torch.bool),
                 reward=torch.zeros(batch_size, length), discount=torch.zeros(batch_size, length),
                 bc_mask=torch.zeros(batch_size, length, dtype=torch.bool),
                 anchor_mask=torch.zeros(batch_size, length, dtype=torch.bool),
                 valid=torch.zeros(batch_size, length, dtype=torch.bool))
    for i, episode in enumerate(episodes):
        n = len(episode['action'])
        for key in ('features', 'action_mask'):
            batch[key][i, :n + 1] = episode[key]
        for key in ('action', 'reward', 'discount'):
            batch[key][i, :n] = episode[key]
        stop_target = episode.get('stop_target', episode['action'] == actions // 2)
        if stop_target.shape != (n,) or stop_target.dtype != torch.bool:
            raise ValueError('STOP targets must be boolean values per real transition')
        batch['stop_target'][i, :n] = stop_target
        bc_mask = episode.get('bc_mask', torch.ones(n, dtype=torch.bool))
        if bc_mask.shape != (n,) or bc_mask.dtype != torch.bool:
            raise ValueError('BC mask must be a boolean value per real transition')
        batch['bc_mask'][i, :n] = bc_mask
        anchor_mask = episode.get('anchor_mask', torch.zeros(n, dtype=torch.bool))
        if anchor_mask.shape != (n,) or anchor_mask.dtype != torch.bool:
            raise ValueError('Anchor mask must be a boolean value per real transition')
        batch['anchor_mask'][i, :n] = anchor_mask
        batch['valid'][i, :n] = True
    return batch
