"""Run from the project root: python experiment/avwan/tests/test_training_pipeline.py."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment.avwan.config.default import get_config
from experiment.avwan.data_pipeline.prepare import prepare, read_episode, angle_sector
from experiment.avwan.data_pipeline.dataset import make_loader
from experiment.avwan.model.hybrid import Network
from experiment.avwan.model.maps import unpack_geometry
from experiment.avwan.training.objectives import objective
from experiment.avwan.training.trainer import train
from experiment.avwan.train import ROOT, load_config, training_config
from experiment.avwan.run import run


def observation(index):
    rng = np.random.default_rng(index)
    gm = np.zeros((400, 400, 2), dtype=np.float32)
    gm[100:150, 180:220, index % 2] = 1
    am = np.zeros((20, 20, 1), dtype=np.float32)
    am[8:12, 8:12] = index + 1
    return dict(rgb=rng.integers(0, 256, (128, 128, 3), dtype=np.uint8),
                depth=np.full((128, 128, 1), index, dtype=np.float32), gm=gm, am=am,
                spectrogram=rng.random((65, 26, 2), dtype=np.float32),
                angle=np.array([99., (index % 8 - 4) * np.pi / 4 + .01], dtype=np.float32))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = load_config(ROOT / 'config/audiogoal.yaml')
        self.config['data'].update(source=str(self.root / 'source'), cache=str(self.root / 'cache'),
                                   shard_size=4, convert_workers=2, val_fraction=.25)
        self.config['training'].update(output=str(self.root / 'run'), workers=0, batch_size=3,
                                       epochs=1, device='cpu', cpu_threads=1, amp=False, log_every=100)
        self.raw, self.previous = {}, {}
        for scene in range(4):
            rows, previous = [], None
            for step in range(3):
                index = scene * 10 + step
                obs = observation(index)
                mask = np.ones(81, dtype=bool)
                mask[0] = False
                rows.append(dict(observation=obs, next_observation=observation(index + 1000),
                                 action=step + 1, action_mask=mask))
                self.raw[index], self.previous[index] = obs, previous
                previous = obs
            path = self.root / 'source' / 'episodes' / f'scene{scene}' / '0.pkl.gz'
            path.parent.mkdir(parents=True)
            with gzip.open(path, 'wb') as handle:
                pickle.dump(dict(format_version=1, transitions=rows), handle)

    def test_history_maps_and_scene_split(self):
        manifest = prepare(self.config)
        seen = {}
        for split in ('train', 'val'):
            ids = set()
            for batch in make_loader(self.config, split):
                self.assertEqual(batch['gm'].shape[1], 40000)
                self.assertNotIn('angle', batch)
                self.assertNotIn('next_observation', batch)
                maps = unpack_geometry(batch['gm'])
                for i, value in enumerate(batch['depth']):
                    index = int(value[0, 0, 0])
                    self.assertNotIn(index, ids)
                    ids.add(index)
                    for key in ('rgb', 'depth', 'spectrogram', 'am'):
                        np.testing.assert_array_equal(batch[key][i].numpy(), self.raw[index][key])
                    np.testing.assert_array_equal(maps[i].numpy(), self.raw[index]['gm'])
                    for key in ('rgb', 'depth'):
                        previous = self.previous[index]
                        expected = previous[key] if previous else np.zeros_like(self.raw[index][key])
                        np.testing.assert_array_equal(batch['prev_' + key][i].numpy(), expected)
                    self.assertEqual(int(batch['angle_label'][i]), angle_sector(self.raw[index]['angle'][1]))
            seen[split] = ids
        self.assertEqual(seen['train'] | seen['val'], set(self.raw))
        self.assertFalse({i // 10 for i in seen['train']} & {i // 10 for i in seen['val']})
        self.assertEqual(prepare(self.config), manifest)
        changed = deepcopy(self.config)
        changed['seed'] += 1
        with self.assertRaises(ValueError):
            prepare(changed)

    def test_nonbinary_map_rejected(self):
        path = next((self.root / 'source').rglob('*.pkl.gz'))
        with gzip.open(path, 'rb') as handle:
            episode = pickle.load(handle)
        episode['transitions'][0]['observation']['gm'][0, 0, 0] = .5
        with gzip.open(path, 'wb') as handle:
            pickle.dump(episode, handle)
        with self.assertRaisesRegex(ValueError, 'binary'):
            read_episode(path, self.config['data']['inputs'])

    def test_persistent_workers_have_complete_deterministic_epochs(self):
        manifest = prepare(self.config)
        self.config['training']['workers'] = 2
        loader = make_loader(self.config, 'train')
        orders = []
        for epoch in (0, 1, 0):
            loader.dataset.set_epoch(epoch)
            ids = [int(x) for batch in loader for x in batch['depth'][:, 0, 0, 0]]
            self.assertEqual(len(ids), manifest['samples']['train'])
            self.assertEqual(len(set(ids)), len(ids))
            orders.append(ids)
        self.assertEqual(orders[0], orders[2])
        self.assertNotEqual(orders[0], orders[1])
        del loader

    def test_new_modalities_receive_gradients_and_mask_invalid_actions(self):
        prepare(self.config)
        batch = next(b for b in make_loader(self.config, 'train') if len(b['action']) == 3)
        model = Network()
        loss, metrics = objective(model, batch, self.config['training'])
        torch.testing.assert_close(loss, metrics['action_loss'] + metrics['angle_loss'])
        loss.backward()
        for weight in (model.geometric_encoder.cnn[0].weight, model.acoustic_encoder.cnn[0].weight,
                       model.mask_encoder[0].weight, model.audio_encoder.fnn[-1].weight,
                       model.visual_encoder.conv2d_1.weight):
            self.assertGreater(float(weight.grad.abs().sum()), 0.)
        model.eval()
        inputs = (batch['spectrogram'], torch.cat((batch['prev_rgb'], batch['rgb']), -1).float() / 255,
                  torch.cat((batch['prev_depth'], batch['depth']), -1), unpack_geometry(batch['gm']),
                  batch['am'], batch['action_mask'])
        with torch.no_grad():
            actions, angles = model.forward_joint(*inputs)
            self.assertEqual(actions.shape, (3, 81))
            self.assertEqual(angles.shape, (3, 8))
            self.assertTrue((actions[:, 0] == -1e9).all())
            self.assertEqual(model.embedding_forward(*inputs).shape, (3, 256))
            first = model.forward_joint(*(x[:1] for x in inputs))
            for one, many in zip(first, (actions, angles)):
                torch.testing.assert_close(one, many[:1], atol=3e-5, rtol=3e-5)
            audio_only = model.audio_encoder(batch['spectrogram'])
            torch.testing.assert_close(angles, audio_only)

    def test_checkpoint_resume_and_export(self):
        manifest = prepare(self.config)
        train(self.config, manifest)
        checkpoint = self.root / 'run' / 'latest.pt'
        first = torch.load(checkpoint, map_location='cpu')
        self.config['training']['epochs'] = 2
        train(self.config, manifest, str(checkpoint))
        second = torch.load(checkpoint, map_location='cpu')
        self.assertEqual(second['epoch'], 2)
        for key in ('geometric_encoder.cnn.0.weight', 'acoustic_encoder.cnn.0.weight', 'mask_encoder.0.weight'):
            self.assertFalse(torch.equal(first['model'][key], second['model'][key]))
        weights = torch.load(self.root / 'run' / 'foundation_weights.pt', weights_only=True)
        Network().load_state_dict(weights)
        self.config['training']['output'] = str(self.root / 'continuous')
        train(self.config, manifest)
        continuous = torch.load(self.root / 'continuous' / 'latest.pt', map_location='cpu')
        for key in second['model']:
            torch.testing.assert_close(second['model'][key], continuous['model'][key], rtol=0, atol=0)

    def test_audiogoal_schema_and_root_training_dispatch(self):
        cfg = get_config()
        self.assertEqual(training_config(cfg.TASK_CONFIG.TRAIN), load_config(ROOT / 'config/audiogoal.yaml'))
        cfg.defrost()
        cfg.TASK_CONFIG.TRAIN.OPEN = True
        cfg.TASK_CONFIG.COLLECT.OPEN = True
        cfg.TASK_CONFIG.TRAIN.EPOCHS = 2
        cfg.freeze()
        with patch('experiment.avwan.train.run_training') as train_run, \
             patch('experiment.avwan.run.COLLECTER') as collect, \
             patch('experiment.AVWANRLEnv') as env, \
             patch('soundspaces.datasets.audionav_dataset.AudioNavDataset') as dataset:
            self.assertEqual(run(cfg), 0)
            self.assertEqual(train_run.call_args.args[0]['training']['epochs'], 2)
            collect.assert_not_called()
            env.assert_not_called()
            dataset.assert_not_called()
        cfg.defrost()
        cfg.TASK_CONFIG.TRAIN.OPEN = cfg.TASK_CONFIG.COLLECT.OPEN = False
        self.assertEqual(run(cfg), 0)
        overrides = self.root / 'overrides.yaml'
        overrides.write_text(yaml.safe_dump({'TASK_CONFIG': {'TRAIN': {'OPEN': True, 'BATCH_SIZE': 7}}}))
        merged = get_config(str(overrides))
        self.assertTrue(merged.TASK_CONFIG.TRAIN.OPEN)
        self.assertEqual(merged.TASK_CONFIG.TRAIN.BATCH_SIZE, 7)


if __name__ == '__main__':
    unittest.main()
