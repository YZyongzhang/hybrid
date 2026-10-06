"""Deploy the exact frozen feature encoder followed by the trained SAC actor."""
from pathlib import Path

import numpy as np
import torch

from .features import load_foundation, sha256
from .sac_cql import RecurrentSACCQL


class OfflinePolicy:
    def __init__(self, foundation, actor, device):
        self.foundation = foundation.eval().requires_grad_(False)
        self.actor = actor.eval().requires_grad_(False)
        self.device = device
        self.reset()

    def reset(self):
        self.previous = None
        self.hidden = None
        if getattr(self, 'action_selector', None):
            self.action_selector.reset()

    @torch.inference_mode()
    def act(self, obs, action_mask):
        previous = self.previous
        rgb = np.concatenate((previous['rgb'] if previous else np.zeros_like(obs['rgb']), obs['rgb']), -1)
        depth = np.concatenate((previous['depth'] if previous else np.zeros_like(obs['depth']), obs['depth']), -1)
        values = [obs['spectrogram'], rgb.astype(np.float32) / 255., depth,
                  obs['gm'], obs['am'], np.asarray(action_mask, dtype=bool)]
        tensors = [torch.as_tensor(np.ascontiguousarray(value), device=self.device).unsqueeze(0) for value in values]
        features = self.foundation.embedding_forward(*tensors)
        selector = getattr(self, 'action_selector', None)
        if selector:
            logits, self.hidden = self.actor.actor(features.unsqueeze(1), self.hidden)
            action = selector(obs, action_mask, logits[0, 0])
        else:
            action, self.hidden = self.actor.act(features, tensors[-1], self.hidden, deterministic=True)
            action = int(action.item())
        if not action_mask[action]:
            raise RuntimeError('SAC actor selected an invalid waypoint')
        self.previous = {key: np.asarray(obs[key]).copy() for key in ('rgb', 'depth')}
        return action

    @classmethod
    def from_checkpoint(cls, state, foundation_path, device):
        signature = state['signature']
        if signature['architecture'] != 'independent_lstm_discrete_sac_cql_v1':
            raise ValueError('Unsupported offline RL checkpoint')
        if sha256(Path(foundation_path)) != signature['features']['checkpoint_sha256']:
            raise ValueError('Foundation checkpoint does not match the training feature cache')
        foundation = load_foundation(foundation_path, device)
        if foundation.action_count != signature['action_count']:
            raise ValueError('Foundation/actor action spaces differ')
        actor = RecurrentSACCQL(signature['feature_dim'], signature['action_count'], state['config']['training']).to(device)
        actor.load_state_dict(state['model'], strict=True)
        return cls(foundation, actor, device)
