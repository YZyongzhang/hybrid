import unittest

import torch

from experiment.avwan.model.audio import instantaneous_binaural_ratio, robust_binaural_ratio
from experiment.avwan.train_source_invariant_foundation import (ReverseGradient, source_view,
                                                                 source_profile_view)


class SourceInvariantAudioTest(unittest.TestCase):
    def test_sound_adversary_reverses_encoder_gradient(self):
        value = torch.ones(3, requires_grad=True)
        ReverseGradient.apply(value, .2).sum().backward()
        self.assertTrue(torch.allclose(value.grad, torch.full_like(value, -.2)))

    def test_robust_ratio_cancels_shared_gain_and_keeps_direction(self):
        torch.manual_seed(17)
        magnitude = torch.rand(3, 65, 26, 2) + .1
        magnitude[..., 0] *= 1.4
        base = robust_binaural_ratio(magnitude.log1p())
        amplified = robust_binaural_ratio((magnitude * 11).log1p())
        self.assertTrue(torch.allclose(base, amplified, atol=2e-6))
        self.assertGreater(float(base[..., 0].mean()), 0)
        self.assertLess(float(base[..., 1].mean()), 0)

    def test_robust_ratio_suppresses_silent_bin_noise(self):
        magnitude = torch.ones(1, 65, 26, 2)
        magnitude[0, 5, 5, 0] = 1e-7
        magnitude[0, 5, 5, 1] = 0
        old = instantaneous_binaural_ratio(magnitude.log1p())
        robust = robust_binaural_ratio(magnitude.log1p())
        self.assertGreater(float(old[0, 5, 5, 0]), .99)
        self.assertLess(float(robust[0, 5, 5, 0]), .001)
        self.assertTrue(torch.isfinite(robust_binaural_ratio(torch.zeros_like(magnitude))).all())

    def test_source_view_preserves_labels_and_masks(self):
        torch.manual_seed(29)
        batch = dict(spectrogram=torch.rand(4, 65, 26, 2),
                     am=torch.ones(4, 20, 20, 1),
                     action=torch.tensor([1, 2, 3, 4]),
                     action_mask=torch.ones(4, 81, dtype=torch.bool))
        view = source_view(batch, strength=1.5, dropout_probability=1.)
        self.assertTrue(torch.equal(view['action'], batch['action']))
        self.assertTrue(torch.equal(view['action_mask'], batch['action_mask']))
        self.assertTrue(torch.isfinite(view['spectrogram']).all())
        blank = view['spectrogram'] == 0
        self.assertTrue(blank.any())
        self.assertTrue(torch.equal(blank[..., 0], blank[..., 1]))

    def test_source_profile_changes_model_input_and_keeps_visible_directions(self):
        torch.manual_seed(31)
        magnitude = torch.rand(4, 65, 26, 2) + .2
        magnitude[..., 0] *= 1.3
        batch = dict(spectrogram=torch.log1p(magnitude),
                     sound_label=torch.tensor([0, 1, 2, 3]),
                     action=torch.tensor([4, 5, 6, 7]))
        swapped = source_profile_view(batch, strength=.7)
        transformed = torch.expm1(swapped['spectrogram'])
        visible = transformed[..., 0] > 0
        before = magnitude[..., 0] / magnitude[..., 1]
        after = transformed[..., 0] / transformed[..., 1].clamp_min(1e-8)
        self.assertTrue(torch.allclose(before[visible], after[visible], rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.equal(swapped['action'], batch['action']))
        original_input = instantaneous_binaural_ratio(batch['spectrogram'])
        augmented_input = instantaneous_binaural_ratio(swapped['spectrogram'])
        self.assertFalse(torch.allclose(original_input, augmented_input))
        self.assertTrue((~visible).any())


if __name__ == '__main__':
    unittest.main()
