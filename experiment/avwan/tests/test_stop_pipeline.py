"""Regression checks for actual failed STOP transitions and selective BC."""
import unittest
import torch
from experiment.avwan.tests.test_offline_rl import small_episode
from experiment.avwan.offline.dataset import collate_episodes
from experiment.avwan.offline.sac_cql import behavior_cloning_loss, RecurrentSACCQL
from experiment.avwan.train_offline_rl import load_config, ROOT


class StopPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_bad_label_excluded_positive_and_movement_retained(self):
        logits = torch.randn(1, 3, 41, requires_grad=True)
        loss = behavior_cloning_loss(logits.log_softmax(-1), torch.tensor([[40, 2, 40]]),
                                     torch.ones(1, 3, dtype=torch.bool), torch.tensor([[False, True, True]]))
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(logits.grad[0, 1:].abs().sum()), 0.)

    def test_all_excluded_finite_zero_gradient(self):
        logits = torch.randn(1, 2, 41, requires_grad=True)
        loss = behavior_cloning_loss(logits.log_softmax(-1), torch.full((1, 2), 40),
                                     torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_collation_and_critic_unchanged(self):
        cfg = load_config(ROOT / 'config/offline_rl.yaml')['training']
        cfg.update(hidden_dim=8, gamma=.9, terminal_mask=True, bc_weight=5.)
        model = RecurrentSACCQL(4, 3, cfg)
        short, long = small_episode(1), small_episode(3)
        batch = collate_episodes([short, long])
        self.assertTrue(torch.equal(batch['bc_mask'], batch['valid']))
        original_action, original_reward = batch['action'].clone(), batch['reward'].clone()
        first = model.losses(batch)
        parameters = tuple(model.q1.parameters()) + tuple(model.q2.parameters())
        g1 = torch.autograd.grad(first['critic_loss'], parameters)
        batch['bc_mask'][0, 0] = False
        second = model.losses(batch)
        g2 = torch.autograd.grad(second['critic_loss'], parameters)
        for key in ['critic_loss', 'td_loss', 'cql_loss', 'td_target']:
            torch.testing.assert_close(first[key], second[key])
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(batch['action'], original_action)
        torch.testing.assert_close(batch['reward'], original_reward)
        self.assertFalse(batch['bc_mask'][0, 1:].any())

    def test_terminal_stop_has_no_bootstrap(self):
        cfg = load_config(ROOT / 'config/offline_rl.yaml')['training']
        cfg.update(hidden_dim=8, gamma=.99, terminal_mask=True)
        model = RecurrentSACCQL(4, 41, cfg)
        ep = dict(features=torch.randn(2, 4), action_mask=torch.ones(2, 41, dtype=torch.bool),
                  action=torch.tensor([40]), reward=torch.tensor([-.01]), discount=torch.tensor([0.]))
        batch = collate_episodes([ep])
        first = model.losses(batch)
        batch['features'][:, 1] = 10000
        second = model.losses(batch)
        torch.testing.assert_close(first['td_target'], torch.tensor(-.01))
        torch.testing.assert_close(first['td_loss'], second['td_loss'])

    def test_invalid_mask_rejected(self):
        ep = small_episode(2)
        ep['bc_mask'] = torch.ones(2)
        with self.assertRaises(ValueError):
            collate_episodes([ep])

    def test_reward_penalty_preserves_loaded_data(self):
        from experiment.avwan.train_stop_rl import annotated_episode
        ep=small_episode(3); original=ep['reward'].clone()
        changed=annotated_episode(ep,[False,False,True],1.)
        torch.testing.assert_close(ep['reward'],original)
        torch.testing.assert_close(changed['reward'],original-torch.tensor([0.,0.,1.]))
        self.assertTrue(torch.equal(ep['action'],changed['action']))
        self.assertTrue(torch.equal(ep['discount'],changed['discount']))

    def test_expert_only_bc_excludes_learner_prefix_and_bad_expert_stop(self):
        from experiment.avwan.train_stop_rl import annotated_episode
        ep=small_episode(4)
        changed=annotated_episode(ep,[False,False,False,True],1.,[False,True,True,True])
        self.assertTrue(torch.equal(changed['bc_mask'],torch.tensor([False,True,True,False])))
        torch.testing.assert_close(changed['action'],ep['action'])
        torch.testing.assert_close(changed['discount'],ep['discount'])


if __name__ == '__main__':
    unittest.main()
