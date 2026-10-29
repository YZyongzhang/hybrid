"""Regression checks for the repaired AVNav experiment, using synthetic/train data only."""
import unittest
import torch
import tempfile
from types import SimpleNamespace
import numpy as np
import json
import gzip
from pathlib import Path
from experiment.avnav.hybrid.preprocessing import normalize_audio, spatial_tokens
from experiment.avnav.hybrid.foundation_model import Network
from trash.train_codex_saccql_lstm import TransitionFeatures, _optimizer_update
from experiment.avnav.offline_rl_v1_5_lstm.OfflineNet import SAC_LSTM_CQL_v1_5
from trash.evaluate_codex_saccql_lstm import _run_episode, _prepare_runtime_dataset_manifest
from trash.run_repaired_pipeline import checked_result
from trash.evaluate_repaired import verify_frozen
from trash.train_codex_foundation import _sha256


def policy(state_dim=260):
    return SAC_LSTM_CQL_v1_5(state_dim, 16, 4, 1e-3, 1e-3, 1e-3, 1.08, .1, .99, .5,
                            torch.device('cpu'), lstm_hidden_dim=16, use_target_encoder=True)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_sequence_padding_previous_action_and_no_episode_leak(self):
        episode = {'features': torch.randn(4, 256), 'actions': torch.tensor([1, 2, 0]),
                   'rewards': torch.tensor([1., 2., 3.]), 'dones': torch.tensor([0., 0., 1.]),
                   'collisions': torch.tensor([True, False, False])}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'train.pt'
            torch.save({'episodes': [episode, episode]}, path)
            ds = TransitionFeatures(path, 5, .5, True, True, 0.)
            states, actions, rewards, following, done, weights = ds.batch(torch.arange(6), 'cpu')
        self.assertEqual(tuple(states.shape), (6, 5, 260))
        self.assertTrue((states[[0, 3], :-1] == 0).all())
        self.assertTrue((states[[0, 3], -1, -4:] == 0).all())
        torch.testing.assert_close(states[1, -1, -4:], torch.tensor([0., 1., 0., 0.]))
        torch.testing.assert_close(following[0], states[1])
        torch.testing.assert_close(rewards[:3], torch.tensor([.5, 2., 3.]))
        torch.testing.assert_close(weights[:3], torch.tensor([0., 1., 1.]))

    def test_update_masks_collision_bc_and_updates_target_encoder(self):
        model = policy()
        before = [p.clone() for p in model.target_temporal_encoder.parameters()]
        states = torch.randn(4, 5, 260)
        result = model.update(states, torch.tensor([1, 1, 1, 1]), torch.ones(4), states + .1,
                              torch.tensor([0., 0., 0., 1.]), bc_weight=.5, sample_bc_weights=torch.zeros(4))
        self.assertEqual(result['bc_loss'], 0.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.target_temporal_encoder.parameters())))
        for old, target, current in zip(before, model.target_temporal_encoder.parameters(), model.temporal_encoder.parameters()):
            torch.testing.assert_close(target, old * .9 + current * .1)
            self.assertIsNone(target.grad)


class RawEvaluationTests(unittest.TestCase):
    def test_manifest_conversion_preserves_authoritative_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'audited.json'
            original = b'{"episodes": []}\n'
            source.write_bytes(original)
            artifact = _prepare_runtime_dataset_manifest('test_fake', root, root / 'run', source)
            self.assertEqual(artifact['source_manifest_sha256'], _sha256(source))
            with gzip.open(artifact['runtime_manifest'], 'rb') as handle:
                self.assertEqual(handle.read(), original)

    def test_collisions_do_not_override_actions_and_history_matches_replay(self):
        obs = {'rgb': np.zeros((2, 2, 3), np.uint8), 'depth': np.zeros((2, 2, 1), np.float32),
               'spectrogram': np.zeros((65, 26, 2), np.float32)}
        sim = SimpleNamespace(previous_step_collided=False, get_agent_state=lambda: SimpleNamespace(position=np.zeros(3)))
        actions, sequences = [], []
        planned = [1, 1, 2, 1, 0]
        def step(action):
            actions.append(action)
            sim.previous_step_collided = action == 1
            return obs, 0., len(actions) == len(planned), {'spl': 0., 'success': False}
        env = SimpleNamespace(reset=lambda: obs, step=step,
                              _env=SimpleNamespace(_sim=sim, current_episode=SimpleNamespace(scene_id='fake.glb', episode_id='1'),
                                                   get_metrics=lambda: {'spl': 0., 'success': False}))
        def get_action(sequence, eval):
            sequences.append(sequence.clone())
            return planned[len(actions)]
        p = SimpleNamespace(temporal_encoder=SimpleNamespace(input_size=260), get_action=get_action)
        foundation = SimpleNamespace(embedding_forward=lambda *args: torch.zeros(1, 256))
        result = _run_episode(env, foundation, p, 'cpu', 5, 200)
        self.assertEqual(actions, planned)
        self.assertEqual(result['collisions'], 3)
        self.assertEqual(result['max_consecutive_collisions'], 2)
        self.assertTrue((sequences[0][0, -1, -4:] == 0).all())
        torch.testing.assert_close(sequences[1][0, -1, -4:], torch.tensor([0., 1., 0., 0.]))

    def test_incomplete_or_modified_artifacts_cannot_pass_selection_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'policy.pt'
            checkpoint.write_bytes(b'fixed-checkpoint')
            result = root / 'val.json'
            base = {'status': 'pass', 'checkpoint_sha256': _sha256(checkpoint),
                    'evaluated_episodes': 2, 'expected_episodes': 2, 'partial_validation_smoke': False,
                    'evaluation_errors': 0, 'missing_expected_ids': 0, 'unexpected_ids': 0}
            result.write_text(json.dumps(base))
            self.assertIsNotNone(checked_result(result, checkpoint))
            for patch in ({'partial_validation_smoke': True}, {'evaluated_episodes': 1}, {'evaluation_errors': 1}):
                result.write_text(json.dumps({**base, **patch}))
                with self.assertRaises(RuntimeError):
                    checked_result(result, checkpoint)
            result.write_text(json.dumps(base))
            frozen = {'immutable_files': {str(checkpoint): _sha256(checkpoint)},
                      'validation_results': [{'path': str(result), 'sha256': _sha256(result)}]}
            verify_frozen(root, frozen)
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                verify_frozen(root, frozen)


class FeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)

    def test_audio_batch_independent_and_silent(self):
        a = torch.rand(1, 65, 26, 2)
        b = torch.rand_like(a) * 100 + 50
        torch.testing.assert_close(normalize_audio(a), normalize_audio(torch.cat([a, b]))[:1])
        self.assertTrue(torch.isfinite(normalize_audio(torch.zeros_like(a))).all())

    def test_spatial_token_corresponds_to_pixel(self):
        x = torch.arange(24).reshape(1, 3, 2, 4)
        for y in range(2):
            for z in range(4):
                torch.testing.assert_close(spatial_tokens(x)[0, y * 4 + z], x[0, :, y, z])

    def test_entire_embedding_batch_independent(self):
        model = Network().cpu().eval()
        a = torch.rand(2, 65, 26, 2)
        a[1] = a[1] * 100 + 50
        rgb = torch.rand(2, 128, 128, 6)
        depth = torch.rand(2, 128, 128, 2)
        with torch.inference_mode():
            single = model.embedding_forward(a[:1], rgb[:1], depth[:1])
            batch = model.embedding_forward(a, rgb, depth)
            torch.testing.assert_close(single, batch[:1], atol=2e-6, rtol=2e-5)
            self.assertEqual(tuple(single.shape), (1, 256))
            action, angle = model.forward_joint(a, rgb, depth)
            self.assertEqual(tuple(action.shape), (2, 4))
            self.assertEqual(tuple(angle.shape), (2, 8))
        self.assertEqual(model.add_position.position_matrix.device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
