import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np

from experiment.avwan.model.audio import binaural_whiten, instantaneous_binaural_ratio
from experiment.avwan.model.hybrid import Network
from experiment.avwan.offline.features import load_foundation
from experiment.avwan.train_unheard import augment_acoustics
from experiment.avwan.revisit_policy import RevisitSelector
from experiment.avwan.ensemble_policy import FoundationEnsemblePolicy
from experiment.avwan.training.objectives import objective


class AcousticDomainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_revisit_uses_observations_preserves_stop_and_resets(self):
        selector = RevisitSelector(1.)
        obs = dict(rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth=np.zeros((2, 2, 1), dtype=np.float32))
        mask = np.array([True, False, True, True, True])
        logits = torch.tensor([3., 100., 2.8, 2.5, 1.])  # center=STOP
        self.assertEqual(selector(obs, mask, logits), 0)
        # Movement alternatives change, but a lower motion score must not
        # accidentally promote STOP above its original decision margin.
        self.assertEqual(selector(obs, mask, logits), 3)
        self.assertEqual(selector(obs, mask, logits+torch.tensor([0., 0., 1., 0., 0.])), 2)
        obs['rgb'][0, 0, 0] = 1
        self.assertEqual(selector(obs, mask, logits), 0)
        selector.reset()
        self.assertEqual(selector.changed, 0)
        self.assertEqual(selector(obs, mask, logits), 0)

    def test_whitening_preserves_ear_ratio_and_removes_gain(self):
        x = torch.rand(3, 65, 26, 2) + .2
        x[..., 0] *= 2
        y = binaural_whiten(x.log1p())
        shifted = binaural_whiten((x*13).log1p())
        self.assertTrue(torch.allclose(y, shifted, atol=1e-6))
        ratio = torch.expm1(y)[..., 0]/torch.expm1(y)[..., 1]
        self.assertTrue(torch.allclose(ratio, x[..., 0]/x[..., 1], atol=2e-6))
        self.assertTrue(torch.isfinite(binaural_whiten(torch.zeros_like(x))).all())

    def test_instantaneous_ratio_cancels_time_varying_source_spectrum(self):
        magnitude=torch.rand(3,65,26,2)+.1
        coloration=torch.exp(torch.randn(3,65,26,1))
        base=instantaneous_binaural_ratio(magnitude.log1p())
        changed=instantaneous_binaural_ratio((magnitude*coloration).log1p())
        torch.testing.assert_close(base,changed,atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(base[...,0],-base[...,1]))
        self.assertTrue(torch.equal(instantaneous_binaural_ratio(torch.zeros_like(magnitude)),torch.zeros_like(magnitude)))

    def test_stop_has_no_undefined_angle_supervision(self):
        class Model:
            action_count=81
            def __init__(self):
                self.actions=torch.randn(1,81,requires_grad=True)
                self.angles=torch.randn(1,8,requires_grad=True)
            def forward_joint(self,*args):return self.actions,self.angles
        model=Model()
        batch=dict(rgb=torch.zeros(1,2,2,3),prev_rgb=torch.zeros(1,2,2,3),
                   depth=torch.zeros(1,2,2,1),prev_depth=torch.zeros(1,2,2,1),
                   spectrogram=torch.zeros(1,2,2,2),gm=torch.zeros(1,400,400,2),
                   am=torch.zeros(1,20,20,1),action_mask=torch.ones(1,81,dtype=torch.bool),
                   action=torch.tensor([40]),angle_label=torch.tensor([7]))
        cfg=dict(action_loss_weight=0.,angle_loss_weight=1.,angle_nonstop_only=True)
        loss,_=objective(model,batch,cfg)
        self.assertEqual(float(loss),0.);loss.backward()
        self.assertTrue((model.angles.grad==0).all())
        batch['action']=torch.tensor([0]);model.angles.grad=None
        loss,_=objective(model,batch,cfg);loss.backward()
        self.assertGreater(float(model.angles.grad.norm()),0.)

    def test_augmentation_keeps_labels_and_binaural_ratios(self):
        x = torch.rand(8, 65, 26, 2) + .1
        batch = dict(spectrogram=x.log1p(), am=torch.rand(8, 20, 20, 1), action=torch.arange(8))
        result = augment_acoustics(batch)
        y = torch.expm1(result['spectrogram'])
        self.assertTrue(torch.allclose(y[..., 0]/y[..., 1], x[..., 0]/x[..., 1], rtol=3e-5))
        self.assertIs(result['action'], batch['action'])
        self.assertTrue(torch.equal(batch['spectrogram'], x.log1p()))
        scaling = result['am']/batch['am']
        self.assertTrue(torch.allclose(scaling, scaling[:, :1, :1, :1].expand_as(scaling)))

    def test_checkpoint_reloads_preprocessing_and_scale_invariant_policy(self):
        options = dict(audio_normalization='binaural_whiten', map_normalization='relative', spatial_detail=True)
        model = Network(**options).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'foundation.pt'
            torch.save(dict(model=model.state_dict(), config=dict(network=options)), path)
            restored = load_foundation(path, 'cpu')
        values = [torch.rand(2, 65, 26, 2).log1p(), torch.rand(2, 128, 128, 6),
                  torch.rand(2, 128, 128, 2), torch.zeros(2, 400, 400, 2),
                  torch.rand(2, 20, 20, 1), torch.ones(2, 81, dtype=torch.bool)]
        with torch.no_grad():
            original = model.forward_joint(*values)[0]
            replay = restored.forward_joint(*values)[0]
            values[0] = (torch.expm1(values[0])*9).log1p()
            values[4] *= 81
            scaled = restored.forward_joint(*values)[0]
        self.assertTrue(torch.equal(original, replay))
        self.assertTrue(torch.allclose(replay, scaled, atol=1e-5))

    def test_spatial_detail_starts_as_identity_and_receives_gradient(self):
        base = Network().eval()
        detail = Network(spatial_detail=True).eval()
        missing = detail.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(all(k.startswith('detail_encoder.') for k in missing.missing_keys))
        values = [torch.rand(1, 65, 26, 2), torch.rand(1, 128, 128, 6),
                  torch.rand(1, 128, 128, 2), torch.zeros(1, 400, 400, 2),
                  torch.rand(1, 20, 20, 1), torch.ones(1, 81, dtype=torch.bool)]
        with torch.no_grad():expected = base.forward_joint(*values)[0]
        actual = detail.forward_joint(*values)[0]
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(actual, torch.tensor([40])).backward()
        self.assertGreater(float(detail.detail_encoder[-1].weight.grad.norm()), 0.)

    def test_ensemble_averages_probabilities_and_preserves_frame_history(self):
        class Member:
            def __init__(self,p):self.logits=torch.tensor([p]).log();self.rgb=None
            def forward_joint(self,audio,rgb,*rest):
                self.rgb=rgb.clone()
                return self.logits, None
        policy=FoundationEnsemblePolicy.__new__(FoundationEnsemblePolicy)
        policy.device='cpu';policy.models=[Member([.8,.19,.01]),Member([.05,.4,.55])]
        policy.weights=torch.tensor([.5,.5]);policy.reset()
        obs=dict(rgb=np.full((2,2,3),100,dtype=np.uint8),depth=np.ones((2,2,1),dtype=np.float32),
                 spectrogram=np.zeros((2,2,2),dtype=np.float32),gm=np.zeros((2,2,2),dtype=np.float32),
                 am=np.zeros((2,2,1),dtype=np.float32))
        self.assertEqual(policy.act(obs,np.ones(3,dtype=bool)),0)
        self.assertTrue((policy.models[0].rgb[...,:3]==0).all())
        self.assertEqual(policy.act(obs,np.array([False,True,True])),1)
        self.assertTrue(torch.allclose(policy.models[0].rgb[...,:3],torch.full((1,2,2,3),100/255)))
        policy.reset();self.assertIsNone(policy.previous)


if __name__ == '__main__':
    unittest.main()
