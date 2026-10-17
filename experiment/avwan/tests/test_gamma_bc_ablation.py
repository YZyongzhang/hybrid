"""Objective isolation and terminal bootstrap regressions for gamma ablation."""
import copy
import unittest
import torch
from experiment.avwan.tests.test_offline_rl import small_episode
from experiment.avwan.offline.dataset import collate_episodes
from experiment.avwan.offline.sac_cql import RecurrentSACCQL
from experiment.avwan.train_offline_rl import load_config, ROOT
from experiment.avwan.gamma_bc_ablation import pure_bc_update, diagnostics


class GammaBCObjectives(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        cfg=load_config(ROOT/'config/offline_rl.yaml')['training']
        cfg.update(hidden_dim=8,gamma=.99,terminal_mask=True,bc_weight=5.,discount_mode='fixed_gamma')
        self.model=RecurrentSACCQL(4,3,cfg)
        self.batch=collate_episodes([small_episode(3),small_episode(1)])

    def test_pure_bc_independent_of_reward_discount_and_critic(self):
        other=copy.deepcopy(self.model);batch=copy.deepcopy(self.batch)
        batch['reward'].fill_(10000);batch['discount'].fill_(.25)
        with torch.no_grad():
            for p in other.q1.parameters():p.add_(100.)
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        pure_bc_update(self.model,self.batch,self.model.make_optimizers()['actor'])
        pure_bc_update(other,batch,other.make_optimizers()['actor'])
        for a,b in zip(self.model.actor.parameters(),other.actor.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for k,v in self.model.state_dict().items():
            if not k.startswith('actor.'):torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in self.model.state_dict().items() if k.startswith('actor.')))

    def test_gamma_zero_ignores_all_future_values_and_terminal_is_masked(self):
        # Constant next-state Q lets the expected soft target be computed exactly.
        with torch.no_grad():
            for head in [self.model.target_q1,self.model.target_q2]:
                for p in head.parameters():p.zero_()
                head.head[-1].bias.fill_(2.)
        self.model.config['gamma']=0.
        a=self.model.losses(self.batch)
        reward=self.batch['reward'][self.batch['valid']].mean()
        torch.testing.assert_close(a['td_target'],reward)
        self.model.config['gamma']=.99
        b=self.model.losses(self.batch)
        self.assertGreater(float(b['td_target']),float(reward))
        self.batch['discount'].zero_()
        c=self.model.losses(self.batch)
        torch.testing.assert_close(c['td_target'],reward)

    def test_smdp_uses_each_transition_discount_and_terminal_zero(self):
        with torch.no_grad():
            for head in [self.model.target_q1,self.model.target_q2]:
                for p in head.parameters():p.zero_()
                head.head[-1].bias.fill_(2.)
        self.model.config['discount_mode']='smdp'
        self.batch['discount'].fill_(.2)
        low=self.model.losses(self.batch)
        self.batch['discount'].fill_(.8)
        high=self.model.losses(self.batch)
        self.assertGreater(float(high['td_target']),float(low['td_target']))
        torch.testing.assert_close(high['discount_used'],torch.tensor(.8))
        self.batch['discount'].zero_()
        terminal=self.model.losses(self.batch)
        torch.testing.assert_close(terminal['td_target'],self.batch['reward'][self.batch['valid']].mean())

    def test_diagnostics_do_not_change_training_update(self):
        other=copy.deepcopy(self.model)
        before=torch.get_rng_state().clone()
        d=diagnostics(self.model,self.batch)
        self.assertGreater(d['weighted_bc_grad_norm'],0.)
        self.assertGreater(d['rl_grad_norm'],0.)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.model.train();other.train()
        self.model.update(self.batch,self.model.make_optimizers())
        other.update(self.batch,other.make_optimizers())
        for k,v in self.model.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k],rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_diagnostic_backward_from_eval_restores_mode(self):
        self.model.cuda().eval()
        batch={k:v.cuda() for k,v in self.batch.items()}
        d=diagnostics(self.model,batch)
        self.assertGreater(d['weighted_bc_grad_norm'],0.)
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))


if __name__=='__main__':unittest.main()
