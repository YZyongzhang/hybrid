"""CPU regression tests; no Habitat/simulator imports are required."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avwan.offline.dataset import collate_episodes
from avwan.offline.features import embed, extract, load_foundation, read_episode
from avwan.offline.sac_cql import RecurrentSACCQL, conservative_gap, masked_policy
from avwan.offline.policy import OfflinePolicy
from avwan.offline.trainer import train
from avwan.train_offline_rl import ROOT, load_config
from avwan.model.hybrid import Network


def observation(index):
    return dict(rgb=np.full((128, 128, 3), index, np.uint8),
                depth=np.full((128, 128, 1), index, np.float32),
                spectrogram=np.full((65, 26, 2), index, np.float32),
                gm=np.zeros((400, 400, 2), np.float32), am=np.zeros((20, 20, 1), np.float32))


def write_episode(path, truncated=False):
    mask = np.array([True, False, True])
    rows = []
    for i in range(2):
        terminal = i == 1
        rows.append(dict(observation=observation(i + 1), next_observation=observation(i + 2),
                         action=2, action_mask=mask, next_action_mask=mask,
                         reward=1., gamma=.9, num_steps=2, discount=.81 if not terminal or truncated else 0.,
                         done=terminal, terminated=terminal and not truncated, truncated=terminal and truncated))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wb') as handle:
        pickle.dump(dict(format_version=1, transitions=rows), handle)


def small_episode(length=3):
    masks = torch.tensor([[True, False, True]]).expand(length + 1, -1).clone()
    return dict(features=torch.randn(length + 1, 4), action_mask=masks,
                action=torch.full((length,), 2, dtype=torch.long), reward=torch.ones(length),
                discount=torch.tensor([.81] * (length - 1) + [0.]))


class OfflineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.config = load_config(ROOT / 'config/offline_rl.yaml')
        self.cfg = self.config['training']
        self.cfg.update(hidden_dim=8, batch_size=2, epochs=2, tensorboard=False)

    def test_pure_bc_actor_is_independent_of_critic_values(self):
        cfg = dict(self.cfg, actor_rl_weight=0., bc_weight=1.)
        model = RecurrentSACCQL(4, 3, cfg)
        batch = collate_episodes([small_episode(4)])
        first = model.losses(batch)
        torch.testing.assert_close(first['actor_loss'], first['policy_nll'])
        grads = torch.autograd.grad(first['actor_loss'], tuple(model.actor.parameters()))
        with torch.no_grad():
            for critic in (model.q1, model.q2):
                for p in critic.parameters():
                    p.mul_(20.)
        second = model.losses(batch)
        changed = torch.autograd.grad(second['actor_loss'], tuple(model.actor.parameters()))
        for a, b in zip(grads, changed):
            torch.testing.assert_close(a, b)

    def test_corrective_labels_leave_behavior_actions_and_rewards_untouched(self):
        from experiment.avwan.train_corrective_memory import actor_bc_loss, relabel_for_bc
        original = small_episode(4)
        corrected = relabel_for_bc(original, torch.zeros(4,dtype=torch.long))
        self.assertTrue((original['action']==2).all())
        self.assertTrue((corrected['action']==0).all())
        model = RecurrentSACCQL(4,3,self.cfg)
        batch = collate_episodes([corrected])
        loss,_ = actor_bc_loss(model,batch)
        batch['reward'].fill_(1e20);batch['discount'].fill_(0.)
        changed,_ = actor_bc_loss(model,batch)
        torch.testing.assert_close(loss,changed)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.q1.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))

    def test_masks_cql_and_temperature_gradient(self):
        logits = torch.tensor([[1., 10000., 2.]], requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        p, lp = masked_policy(logits, mask)
        self.assertEqual(float(p[0, 1]), 0.)
        self.assertTrue(torch.isfinite(lp).all())
        gap = conservative_gap(logits, torch.tensor([2]), mask, 1.)
        torch.testing.assert_close(gap, torch.logsumexp(logits[:, [0, 2]], -1) - 2.)
        gap.sum().backward()
        self.assertEqual(float(logits.grad[0, 1]), 0.)
        model = RecurrentSACCQL(4, 3, self.cfg)
        with torch.no_grad():
            model.actor.head[-1].weight.zero_()
            model.actor.head[-1].bias.copy_(torch.tensor([50., 0., -50.]))
        loss = model.losses(collate_episodes([small_episode()]))
        loss['alpha_loss'].backward()
        self.assertLess(float(model.log_alpha.grad), 0.)  # optimizer raises alpha

    def test_causal_history_and_padding(self):
        model = RecurrentSACCQL(4, 3, self.cfg)
        episode = small_episode(3)
        model.eval()
        logits, _ = model.actor(episode['features'][None])
        hidden, outputs = None, []
        for feature in episode['features']:
            out, hidden = model.actor(feature[None, None], hidden)
            outputs.append(out)
        torch.testing.assert_close(logits, torch.cat(outputs, dim=1))
        short, long = small_episode(1), small_episode(4)
        batch = collate_episodes([short, long])
        first = model.losses(batch)
        batch['features'][0, 2:] = 999.
        batch['reward'][0, 1:] = 777.
        second = model.losses(batch)
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    def test_training_gamma_target_and_optimizer_boundaries(self):
        self.cfg['gamma'] = .7
        model = RecurrentSACCQL(4, 3, self.cfg)
        with torch.no_grad():
            for network in (model.target_q1, model.target_q2):
                for parameter in network.parameters():
                    parameter.zero_()
                network.head[-1].bias.fill_(2.)
            for parameter in model.actor.parameters():
                parameter.zero_()
        episode = small_episode(2)
        episode['discount'][:] = torch.tensor([0., .81])
        batch = collate_episodes([episode])
        loss = model.losses(batch)
        expected = 1. + self.cfg['gamma'] * (2. + .1 * np.log(2.))
        self.assertAlmostEqual(float(loss['td_target']), expected, places=5)
        batch['discount'].fill_(.2)
        torch.testing.assert_close(model.losses(batch)['td_target'], loss['td_target'])
        self.cfg['gamma'] = 0.
        self.assertAlmostEqual(float(model.losses(batch)['td_target']), 1., places=5)
        self.cfg['gamma'] = .7
        actor_before = deepcopy(model.actor.state_dict())
        target_before = deepcopy(model.target_q1.state_dict())
        model.update(batch, model.make_optimizers())
        self.assertTrue(any(not torch.equal(v, actor_before[k]) for k, v in model.actor.state_dict().items()))
        for key, value in model.target_q1.state_dict().items():
            torch.testing.assert_close(value, target_before[key].lerp(model.q1.state_dict()[key], self.cfg['tau']))
        self.assertTrue(all(p.grad is None for p in model.target_q1.parameters()))
        action, _ = model.act(episode['features'][:1], episode['action_mask'][:1])
        self.assertNotEqual(int(action), 1)

    def test_optional_terminal_mask_preserves_timeout_bootstrap(self):
        self.cfg.update(gamma=.7, terminal_mask=True)
        model = RecurrentSACCQL(4, 3, self.cfg)
        with torch.no_grad():
            for network in (model.target_q1, model.target_q2):
                for parameter in network.parameters():
                    parameter.zero_()
                network.head[-1].bias.fill_(2.)
            for parameter in model.actor.parameters():
                parameter.zero_()
        episode = small_episode(2)
        episode['discount'][:] = torch.tensor([0., .81])
        batch = collate_episodes([episode])
        expected = 1. + .5 * .7 * (2. + .1 * np.log(2.))
        self.assertAlmostEqual(float(model.losses(batch)['td_target']), expected, places=5)
        batch['discount'].zero_()
        self.assertAlmostEqual(float(model.losses(batch)['td_target']), 1., places=5)

    def test_bc_regularization_actor_gradient(self):
        model = RecurrentSACCQL(4, 3, self.cfg)
        batch = collate_episodes([small_episode(2)])
        original = model.losses(batch)
        self.cfg['bc_weight'] = 2.
        regularized = model.losses(batch)
        torch.testing.assert_close(regularized['actor_loss'], original['actor_loss'] + 2 * original['policy_nll'])
        regularized['actor_loss'].backward()
        self.assertGreater(float(model.actor.lstm.weight_hh_l0.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.q1.parameters()))

    def test_residual_actor_starts_at_frozen_prior(self):
        self.cfg['foundation_prior'] = True
        model = RecurrentSACCQL(4, 3, self.cfg)
        batch = collate_episodes([small_episode(3)])
        with torch.no_grad():
            logits, _ = model.actor(batch['features'])
            torch.testing.assert_close(logits, model.actor.prior(batch['features']))
        before = deepcopy(model.actor.prior.state_dict())
        model.update(batch, model.make_optimizers())
        for key, value in model.actor.prior.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_frozen_extraction_history_cache_and_resume_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = deepcopy(self.config)
            cfg['features'].update(checkpoint=str(root / 'foundation.pt'), source=str(root / 'source'),
                                   val_source=str(root / 'val'), cache=str(root / 'cache'),
                                   batch_size=2, workers=1, episode_batch_size=1)
            for split in ('source', 'val'):
                write_episode(root / split / 'episodes' / split / '0.pkl.gz', truncated=split == 'val')
            torch.save(Network(3).state_dict(), cfg['features']['checkpoint'])
            model = load_foundation(cfg['features']['checkpoint'], torch.device('cpu'))
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            packed, arrays = read_episode(root / 'source/episodes/source/0.pkl.gz', 3)

            class Probe:
                def embedding_forward(self, audio, rgb, depth, gm, am, mask):
                    return depth[:, 0, 0, :]

            history = embed(Probe(), [(packed, arrays), (packed, arrays)], 3, 'cpu')
            for episode in history:
                np.testing.assert_array_equal(episode['features'], [[0., 1.], [1., 2.], [2., 3.]])
            before = deepcopy(model.state_dict())
            first = embed(model, [(packed, arrays)], 1, 'cpu')[0]['features']
            second = embed(model, [(packed, arrays)], 3, 'cpu')[0]['features']
            np.testing.assert_allclose(first, second, atol=2e-5, rtol=2e-5)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
            actor = RecurrentSACCQL(256, 3, self.cfg).eval()
            deployed = OfflinePolicy(model, actor, 'cpu')
            cached_hidden = None
            for i in range(2):
                predicted, cached_hidden = actor.act(torch.from_numpy(first[i:i+1]),
                                                     torch.from_numpy(arrays['action_mask'][i:i+1]), cached_hidden)
                self.assertEqual(deployed.act(observation(i + 1), arrays['action_mask'][i]), int(predicted))
                for actual, expected in zip(deployed.hidden, cached_hidden):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            deployed.reset()
            self.assertIsNone(deployed.hidden)
            self.assertIsNone(deployed.previous)
            manifest = extract(cfg, torch.device('cpu'))
            self.assertEqual(manifest['transitions'], {'train': 2, 'val': 2})
            self.assertEqual(extract(cfg, torch.device('cpu')), manifest)
            # Simulate interruption after all per-episode files but before the final marker.
            (root / 'cache/manifest.json').unlink()
            self.assertEqual(extract(cfg, torch.device('cpu')), manifest)
            cfg['training']['output'] = str(root / 'uninterrupted')
            train(cfg, manifest, torch.device('cpu'))
            cfg['training'].update(output=str(root / 'resumed'), epochs=1)
            train(cfg, manifest, torch.device('cpu'))
            cfg['training']['epochs'] = 2
            train(cfg, manifest, torch.device('cpu'), str(root / 'resumed/latest.pt'))
            full = torch.load(root / 'uninterrupted/latest.pt', weights_only=False)
            resumed = torch.load(root / 'resumed/latest.pt', weights_only=False)
            self.assertEqual(full['global_step'], resumed['global_step'])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            steps = [json.loads(line) for line in (root / 'resumed/steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in steps], [1, 2])
            self.assertTrue({'entropy', 'actor_loss', 'critic_loss', 'alpha'} <= steps[-1].keys())
            self.assertTrue((root / 'resumed/best_policy.pt').exists())
            self.assertEqual(len(json.loads((root / 'resumed/top_checkpoints.json').read_text())), 3)
            deployed = OfflinePolicy.from_checkpoint(full, cfg['features']['checkpoint'], 'cpu')
            self.assertIsNone(deployed.hidden)
            bad_state = deepcopy(full)
            bad_state['signature']['features']['checkpoint_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'does not match'):
                OfflinePolicy.from_checkpoint(bad_state, cfg['features']['checkpoint'], 'cpu')
            cfg['training']['cql_weight'] += 1
            with self.assertRaisesRegex(ValueError, 'hyperparameter'):
                train(cfg, manifest, torch.device('cpu'), str(root / 'resumed/latest.pt'))
            cfg['features']['max_episodes'] = 1
            with self.assertRaisesRegex(ValueError, 'changed'):
                extract(cfg, torch.device('cpu'))

    def test_rejects_broken_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode.pkl.gz'
            write_episode(path, truncated=True)
            _, arrays = read_episode(path, 3)
            self.assertAlmostEqual(float(arrays['discount'][-1]), .81)
            with gzip.open(path, 'rb') as handle:
                episode = pickle.load(handle)
            episode['transitions'][1]['observation']['depth'] = np.zeros((128, 128, 1), np.float32)
            with gzip.open(path, 'wb') as handle:
                pickle.dump(episode, handle)
            with self.assertRaisesRegex(ValueError, 'noncontiguous'):
                read_episode(path, 3)


if __name__ == '__main__':
    unittest.main()
