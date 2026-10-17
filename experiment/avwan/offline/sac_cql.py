"""Discrete SAC-CQL with independent actor/twin-critic LSTMs.

Reference: SoundSpaces AVNav's yz/net/sac_cql/trainer/SAC_CQL.py.
Adaptations: 81 masked waypoints, configurable training gamma, consistent recurrent
histories, target recurrent critics, and a positive mask-aware entropy target.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_policy(logits, mask):
    if not mask.any(-1).all():
        raise ValueError('Every state needs at least one valid action')
    log_probs = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    # Mask log-probabilities AFTER normalization to avoid 0 * -inf.
    return log_probs.exp(), log_probs.masked_fill(~mask, 0.)


def mean_valid(values, valid):
    return values[valid].mean()


def behavior_cloning_loss(log_probs, actions, valid, bc_mask=None):
    """Exclude known bad behavior labels without changing critic transitions."""
    nll = -log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    allowed = valid if bc_mask is None else valid & bc_mask.bool()
    # An all-negative batch must have finite zero BC and zero BC gradients.
    return nll[allowed].sum() / allowed.sum().clamp_min(1)


def conservative_gap(q, actions, mask, temperature):
    taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    partition = temperature * torch.logsumexp((q / temperature).masked_fill(~mask, -torch.inf), -1)
    return partition - taken


class RecurrentHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_count, layers, foundation_prior=False,
                 factorized_stop_head=False):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, layers, batch_first=True)
        self.action_count = action_count
        self.stop_action = action_count // 2
        self.factorized_stop_head = bool(factorized_stop_head)
        if self.factorized_stop_head:
            if action_count % 2 != 1:
                raise ValueError('A factorized STOP head requires an odd action count with a center STOP action')
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.motion_head = nn.Linear(hidden_dim, action_count - 1)
            self.stop_head = nn.Linear(hidden_dim, 1)
        else:
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, action_count))
        self.prior = None
        if foundation_prior:
            self.prior = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, action_count))
            self.prior.requires_grad_(False)
            if self.factorized_stop_head:
                nn.init.zeros_(self.motion_head.weight)
                nn.init.zeros_(self.motion_head.bias)
                nn.init.zeros_(self.stop_head.weight)
                nn.init.zeros_(self.stop_head.bias)
            else:
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, hidden=None):
        latent, hidden = self.lstm(features, hidden)
        if self.factorized_stop_head:
            shared = self.head(latent)
            motion_logits = self.motion_head(shared)
            stop_logits = self.stop_head(shared)
            logits = torch.cat((motion_logits[..., :self.stop_action], stop_logits,
                                motion_logits[..., self.stop_action:]), dim=-1)
        else:
            logits = self.head(latent)
        if self.prior is not None:
            logits = self.prior(features) + logits
        return logits, hidden


class RecurrentSACCQL(nn.Module):
    def __init__(self, feature_dim, action_count, config):
        super().__init__()
        self.config = config
        args = (feature_dim, config['hidden_dim'], action_count, config['lstm_layers'])
        self.factorized_stop_head = bool(config.get('factorized_stop_head', False))
        self.stop_action = action_count // 2
        self.actor = RecurrentHead(*args, foundation_prior=config.get('foundation_prior', False),
                                   factorized_stop_head=self.factorized_stop_head)
        self.q1, self.q2 = RecurrentHead(*args), RecurrentHead(*args)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(config['initial_alpha'])))

    def make_optimizers(self):
        lr = self.config['learning_rate']
        weight_decay = self.config.get('weight_decay', 0.)
        return dict(actor=torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay),
                    critic=torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()),
                                            lr=lr, weight_decay=weight_decay),
                    alpha=torch.optim.Adam([self.log_alpha], lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def act(self, feature, action_mask, hidden=None, deterministic=True):
        """One step, [B,F] -> [B] actions and recurrent state; reset hidden per episode."""
        logits, hidden = self.actor(feature.unsqueeze(1), hidden)
        probs, _ = masked_policy(logits[:, 0], action_mask)
        action = probs.argmax(-1) if deterministic else torch.distributions.Categorical(probs).sample()
        return action, hidden

    def losses(self, batch, reference_actor=None):
        states, masks, valid = batch['features'], batch['action_mask'], batch['valid']
        alpha = self.log_alpha.exp().detach()
        # All networks start from zero at the real episode start. Processing
        # [s0,...,sT] ONCE means next values retain s0 in their history.
        logits, _ = self.actor(states)
        probs, log_probs = masked_policy(logits, masks)
        with torch.no_grad():
            target1, _ = self.target_q1(states)
            target2, _ = self.target_q2(states)
            values = (probs.detach() * (torch.minimum(target1, target2) - alpha * log_probs.detach())).sum(-1)
            bootstrap = values[:, 1:]
            discount_mode = self.config.get('discount_mode', 'smdp')
            if discount_mode == 'smdp':
                # Each waypoint transition spans `num_steps` primitive actions.
                # The collector already stores gamma**num_steps, with zero for
                # true terminals and a positive value for time-limit truncation.
                discount = batch['discount']
            elif discount_mode == 'fixed_gamma':
                # Kept only for reproducing older runs that used one decision
                # discount regardless of macro-action duration.
                discount = self.config['gamma']
                if self.config.get('terminal_mask', False):
                    bootstrap = bootstrap * (batch['discount'] > 0)
            else:
                raise ValueError(f'Unknown discount_mode: {discount_mode!r}')
            td_target = batch['reward'] + discount * bootstrap
        q1, _ = self.q1(states[:, :-1])
        q2, _ = self.q2(states[:, :-1])
        action = batch['action']
        taken1 = q1.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        taken2 = q2.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        td = mean_valid(0.5 * ((taken1 - td_target).square() + (taken2 - td_target).square()), valid)
        temperature = self.config['cql_temperature']
        cql = mean_valid(0.5 * (conservative_gap(q1, action, masks[:, :-1], temperature)
                               + conservative_gap(q2, action, masks[:, :-1], temperature)), valid)
        critic_loss = td + self.config['cql_weight'] * cql
        p, lp = probs[:, :-1], log_probs[:, :-1]
        actor_loss = mean_valid((p * (alpha * lp - torch.minimum(q1, q2).detach())).sum(-1), valid)
        policy_nll = mean_valid(-lp.gather(-1, action.unsqueeze(-1)).squeeze(-1), valid)
        bc_nll = behavior_cloning_loss(lp, action, valid, batch.get('bc_mask'))
        actor_loss = (self.config.get('actor_rl_weight', 1.) * actor_loss
                      + self.config.get('bc_weight', 0.) * bc_nll)
        stop_aux_loss = actor_loss.new_zeros(())
        stop_aux_weight = float(self.config.get('stop_aux_weight', 0.))
        if stop_aux_weight > 0:
            stop_target = batch.get('stop_target', action == self.stop_action).bool()
            stop_probability = p[..., self.stop_action].clamp(1e-7, 1. - 1e-7)
            positive = valid & stop_target
            negative = valid & ~stop_target
            positive_nll = -stop_probability.log()
            negative_nll = -torch.log1p(-stop_probability)
            positive_mean = positive_nll[positive].mean() if positive.any() else actor_loss.new_zeros(())
            negative_mean = negative_nll[negative].mean() if negative.any() else actor_loss.new_zeros(())
            stop_aux_loss = 0.5 * (positive_mean + negative_mean)
            actor_loss = actor_loss + stop_aux_weight * stop_aux_loss
        parent_kl = actor_loss.new_zeros(())
        anchor_weight = self.config.get('parent_anchor_weight', 0.)
        anchor_mask = batch.get('anchor_mask')
        if reference_actor is not None and anchor_weight > 0 and anchor_mask is not None:
            with torch.no_grad():
                reference_logits, _ = reference_actor(states[:, :-1])
                reference_probs, reference_log_probs = masked_policy(reference_logits, masks[:, :-1])
            per_state_kl = (reference_probs * (reference_log_probs - lp)).sum(-1)
            allowed_anchor = valid & anchor_mask.bool()
            parent_kl = per_state_kl[allowed_anchor].sum() / allowed_anchor.sum().clamp_min(1)
            actor_loss = actor_loss + anchor_weight * parent_kl
        entropy = -(p * lp).sum(-1)
        target_entropy = self.config['entropy_ratio'] * masks[:, :-1].sum(-1).float().log()
        # If H < target, gradient descent INCREASES alpha.
        alpha_loss = self.log_alpha * mean_valid(entropy.detach() - target_entropy, valid)
        metrics = dict(critic_loss=critic_loss, td_loss=td, cql_loss=cql, actor_loss=actor_loss,
                       alpha_loss=alpha_loss, alpha=alpha, entropy=mean_valid(entropy, valid),
                       target_entropy=mean_valid(target_entropy, valid),
                       action_accuracy=mean_valid((p.argmax(-1) == action).float(), valid),
                       policy_nll=policy_nll, bc_nll=bc_nll,
                       stop_aux_loss=stop_aux_loss,
                       parent_kl=parent_kl,
                       discount_used=mean_valid(
                           discount if isinstance(discount, torch.Tensor)
                           else torch.full_like(batch['reward'], float(discount)), valid),
                       q_data=mean_valid(0.5 * (taken1 + taken2), valid),
                       td_target=mean_valid(td_target, valid))
        return metrics

    def update(self, batch, optimizers, reference_actor=None):
        losses = self.losses(batch, reference_actor=reference_actor)
        if any(not torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError('Nonfinite SAC-CQL loss')
        for key, loss in [('critic', 'critic_loss'), ('actor', 'actor_loss'), ('alpha', 'alpha_loss')]:
            optimizer = optimizers[key]
            optimizer.zero_grad(set_to_none=True)
            losses[loss].backward()
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            nn.utils.clip_grad_norm_(parameters, self.config['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-10., 5.)
            for target, online in [(self.target_q1, self.q1), (self.target_q2, self.q2)]:
                for target_parameter, parameter in zip(target.parameters(), online.parameters()):
                    target_parameter.lerp_(parameter, self.config['tau'])
        return {key: float(value.detach()) for key, value in losses.items()}
