"""Discourage repeating failed visual-state/action pairs using observations only."""
from collections import Counter
import hashlib

import numpy as np
import torch


class RevisitSelector:
    def __init__(self, penalty):
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError('Revisit penalty must be finite and positive')
        self.penalty = float(penalty)
        self.reset()

    def reset(self):
        self.visits = Counter()
        self.changed = 0

    def __call__(self, obs, mask, logits):
        # Sensor images identify actual position/orientation; action IDs alone
        # cannot distinguish progress from repeating a blocked waypoint.
        digest = hashlib.blake2b(digest_size=16)
        for key in ['rgb', 'depth']:
            digest.update(np.ascontiguousarray(obs[key]).tobytes())
        state = digest.digest()
        values = logits.detach().float().clone().reshape(-1)
        valid = torch.as_tensor(mask, dtype=torch.bool, device=values.device)
        values[~valid] = -torch.inf
        stop = len(values)//2
        moving = valid.clone();moving[stop] = False
        original = int(values.argmax())
        if moving.any():
            best = values[moving].max()
            for action in torch.where(moving)[0].tolist():
                values[action] -= self.penalty*self.visits[(state, action)]
            # Preserve the STOP-vs-movement margin; only re-rank motion actions.
            values[moving] += best-values[moving].max()
        chosen = int(values.argmax())
        self.visits[(state, chosen)] += 1
        self.changed += chosen != original
        return chosen
