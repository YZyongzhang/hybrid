"""Numerical memory diagnostics on real cached validation episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .offline.dataset import FeatureEpisodes, collate_episodes
from .offline.sac_cql import RecurrentSACCQL, masked_policy


def audit(checkpoint, cache, output):
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    sig = state['signature']
    model = RecurrentSACCQL(sig['feature_dim'], sig['action_count'], state['config']['training']).eval()
    model.load_state_dict(state['model'])
    manifest = json.loads((Path(cache) / 'manifest.json').read_text())
    episodes = FeatureEpisodes(cache, manifest, 'val')
    errors, differences, decisions, hidden_norms = [], [], [], []
    full_nll, reset_nll = [], []
    with torch.no_grad():
        for ep in episodes:
            x = ep['features'][None]
            full, _ = model.actor(x)
            hidden, streamed = None, []
            for f in ep['features']:
                y, hidden = model.actor(f[None, None], hidden)
                streamed.append(y)
                hidden_norms.append(float(hidden[0].norm()))
            errors.append(float((full - torch.cat(streamed, 1)).abs().max()))
            # Treat each timestep as its own sequence, removing memory only.
            reset, _ = model.actor(ep['features'][:, None])
            reset = reset[:, 0][None]
            mask = ep['action_mask'][None]
            p, lp = masked_policy(full, mask)
            rp, rlp = masked_policy(reset, mask)
            differences.extend((full[:, 1:] - reset[:, 1:]).abs().flatten().tolist())
            decisions.extend((p[:, 1:-1].argmax(-1) != rp[:, 1:-1].argmax(-1)).flatten().tolist())
            actions = ep['action'][None, :, None]
            full_nll.extend((-lp[:, :-1].gather(-1, actions)).flatten().tolist())
            reset_nll.extend((-rlp[:, :-1].gather(-1, actions)).flatten().tolist())
            first, _ = model.actor(x[:, :1], None)
            torch.testing.assert_close(first, full[:, :1], atol=1e-5, rtol=1e-5)
        # Ensure the first real episode actually has a padded suffix to perturb.
        shortest = min(episodes.episodes, key=lambda e: len(e['action']))
        longest = max(episodes.episodes, key=lambda e: len(e['action']))
        batch = collate_episodes([shortest, longest])
        before = model.losses(batch)
        n = len(shortest['action'])
        padding_steps = batch['features'].shape[1] - n - 1
        if padding_steps < 1:
            raise ValueError('Real validation episodes need unequal lengths for the padding audit')
        batch['features'][0, n + 1:] = 10000.
        after = model.losses(batch)
        padding_error = max(float((before[k] - after[k]).abs()) for k in before)
    model.zero_grad()
    ep = max(episodes.episodes, key=lambda e: len(e['action']))
    features = ep['features'][None].clone().requires_grad_(True)
    logits, _ = model.actor(features)
    logits[0, min(3, logits.shape[1]-1)].square().mean().backward()
    gradient = float(features.grad[0, 0].norm())
    recurrent_gradient = float(model.actor.lstm.weight_hh_l0.grad.norm())
    result = dict(checkpoint=str(checkpoint), episodes=len(episodes),
                  streaming_max_abs_error=max(errors), padding_loss_max_abs_error=padding_error,
                  padding_states_perturbed=padding_steps,
                  hidden_norm_mean=float(np.mean(hidden_norms)), history_logit_mean_abs_effect=float(np.mean(differences)),
                  history_action_change_fraction=float(np.mean(decisions)), full_history_policy_nll=float(np.mean(full_nll)),
                  reset_every_step_policy_nll=float(np.mean(reset_nll)), first_state_gradient_norm=gradient,
                  recurrent_weight_gradient_norm=recurrent_gradient,
                  episode_reset_verified=True, memory_is_functional=bool(max(errors)<1e-4 and padding_error<1e-5 and gradient>0 and recurrent_gradient>0))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.checkpoint, args.cache, args.output)
