"""Small CPU integration tests using actual Network, Train and launcher."""
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

from configs.default import get_config
from Dataset.dataset_avnav import AVNavDataset, ShardBatchSampler, convert_pickle_to_pt
from experiment.avnav.run import build_datasets
from experiment.avnav.train_class.hybrid_train import angle_to_sector_label

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        for episode in range(3):
            observations = [dict(rgb=np.full((128, 128, 3), step * 50 + episode, dtype=np.uint8),
                                 depth=np.full((128, 128, 1), step + 1, dtype=np.float32),
                                 spectrogram=np.ones((1, 65, 26, 2), dtype=np.float32),
                                 angle=np.array([1.0, np.pi / 2])) for step in range(3)]
            with (self.raw / f"{episode}.pkl").open("wb") as handle:
                pickle.dump(dict(obs=observations, action_id=[[1, 2]], sound_id=["telephone"]), handle)

    def test_conversion_and_dataset(self):
        destination = self.root / "pt"
        files = convert_pickle_to_pt(self.raw, destination, shard_size=3, shuffle=False)
        self.assertEqual(len(files), 3)  # Never split an episode to fill a shard.
        dataset = AVNavDataset(destination, cache_size=1)
        self.assertEqual(len(dataset), 6)
        audio, rgb, depth, angle, action = dataset[0]
        self.assertEqual(tuple(audio.shape), (65, 26, 2))
        self.assertTrue(torch.equal(rgb[..., :3], torch.zeros_like(rgb[..., :3])))
        self.assertTrue(torch.equal(depth[..., :1], torch.zeros_like(depth[..., :1])))
        self.assertAlmostEqual(angle.item(), 90.0)
        self.assertEqual(action.item(), 1)
        self.assertTrue(torch.equal(dataset[1][1][..., :3], rgb[..., 3:]))
        self.assertTrue(torch.equal(dataset[2][1][..., :3], torch.zeros_like(rgb[..., :3])))
        self.assertEqual(len(dataset._cache), 1)
        self.assertEqual(dataset[-1][-1].item(), 2)
        with self.assertRaises(FileExistsError):
            convert_pickle_to_pt(self.raw, destination)
        cfg = get_config().TASK_CONFIG.TRAIN
        cfg.DATA_PATH = str(destination)
        train, val = build_datasets(cfg, 42)
        self.assertFalse(set(train.files) & set(val.files))
        self.assertEqual(len(train) + len(val), 6)
        cfg.VAL_DATA_PATH = str(destination)
        with self.assertRaises(ValueError):
            build_datasets(cfg, 42)

    def test_launcher_conversion_training_and_checkpoint_loading(self):
        from experiment.avwan.model.hybrid import Network
        destination, checkpoints = self.root / "pt", self.root / "checkpoints"
        config = {"TASK_CONFIG": {
            "PT": {"FROM_PATH": str(self.raw), "TO_PATH": str(destination), "SHARD_SIZE": 2},
            "TRAIN": {"DATA_PATH": str(destination), "EPOCHS": 1, "BATCH_SIZE": 2,
                      "DEVICE": "cpu", "SAVE_DIR": str(checkpoints)}}}
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        import os
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        result = subprocess.run([sys.executable, str(ROOT / "run_avnav.py"), "--config", str(path),
                                 "--stage", "all"], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        weights = torch.load(checkpoints / "model_epoch_1.pth", weights_only=True)
        self.assertTrue(all(torch.isfinite(value).all() for value in weights.values()))
        model = Network()
        model.load_state_dict(weights)
        config["TASK_CONFIG"]["TRAIN"]["MODEL_PATH"] = str(checkpoints / "model_epoch_1.pth")
        config["TASK_CONFIG"]["TRAIN"]["SAVE_DIR"] = str(self.root / "reloaded")
        path.write_text(yaml.safe_dump(config))
        result = subprocess.run([sys.executable, str(ROOT / "run_avnav.py"), "--config", str(path),
                                 "--stage", "train"], cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / "reloaded/model_epoch_1.pth").exists())
        self.assertTrue((checkpoints / "best.pth").exists())
        import json
        history = json.loads((checkpoints / "metrics.json").read_text())
        self.assertEqual(history[0]["train"]["samples"], 4)
        self.assertEqual(history[0]["val"]["samples"], 2)

    def test_collector_audio_tuple_and_shard_sampler(self):
        file = self.raw / "0.pkl"
        with file.open("rb") as handle:
            data = pickle.load(handle)
        for obs in data["obs"]:
            obs["spectrogram"] = (obs["spectrogram"][0], np.zeros((2, 16000)))
        with file.open("wb") as handle:
            pickle.dump(data, handle)
        destination = self.root / "pt"
        convert_pickle_to_pt(self.raw, destination, shard_size=3, shuffle=False)
        dataset = AVNavDataset(destination)
        self.assertEqual(tuple(dataset[0][0].shape), (65, 26, 2))
        sampler = ShardBatchSampler(dataset, 1, seed=42)
        batches = list(sampler)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(batches, list(ShardBatchSampler(dataset, 1, seed=42)))
        self.assertNotEqual(batches, list(sampler))
        for a, b in zip(batches[::2], batches[1::2]):
            self.assertEqual(a[0] // 2, b[0] // 2)

    def test_sector_boundaries(self):
        angles = torch.tensor([-180, -135, 0, 179, 180, 540], dtype=torch.float32)
        self.assertEqual(angle_to_sector_label(angles).tolist(), [0, 1, 4, 7, 0, 0])


if __name__ == "__main__":
    unittest.main()
