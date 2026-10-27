"""AVNav inference must execute the policy's raw argmax action."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiment.avnav.evaluate_online import OnlinePolicy, evaluate_split


class RawActionPolicyTest(unittest.TestCase):
    def _observation(self):
        return {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.zeros((2, 2, 1), dtype=np.float32),
            "spectrogram": np.zeros((1, 65, 26, 2), dtype=np.float32),
        }

    def test_stop_argmax_is_executed_without_threshold(self):
        foundation = Mock()
        foundation.eval.return_value = foundation
        foundation.embedding_forward.return_value = torch.zeros((1, 256))
        actor = Mock()
        actor.eval.return_value = actor
        actor.return_value = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        policy = OnlinePolicy(foundation, actor, torch.device("cpu"))

        action = policy.act(self._observation())

        self.assertEqual(action, 0)
        foundation.embedding_forward.assert_called_once()
        actor.assert_called_once()

    def test_collision_observation_does_not_replace_policy_action(self):
        class FakeEnv:
            def __init__(self, config, dataset):
                self._env = Mock()
                self._env.current_episode = dataset.episodes[0]
                self._env._sim = Mock(previous_step_collided=True)
                self.steps = []

            def reset(self):
                return observation

            def step(self, action):
                self.steps.append(action)
                return observation, 0.0, len(self.steps) == 2, {"spl": 0, "success": False}

            def close(self):
                pass

        observation = self._observation()
        episode = SimpleNamespace(scene_id="scene.glb", episode_id="1")
        dataset = SimpleNamespace(episodes=[episode])
        policy = Mock()
        policy.act.return_value = 1
        env = FakeEnv(None, dataset)
        settings = {"env_config": "unused", "episodes_per_split": 1, "max_steps": 2}
        config = SimpleNamespace(
            TASK_CONFIG=SimpleNamespace(
                TRAIN=SimpleNamespace(OPEN=True),
                COLLECT=SimpleNamespace(OPEN=True),
                TASK=SimpleNamespace(SENSORS=[], MEASUREMENTS=[]),
                DATASET=SimpleNamespace(),
            ),
            defrost=Mock(),
            freeze=Mock(),
        )
        with tempfile.TemporaryDirectory() as output:
            with patch("experiment.avnav.environment.avnav_env.AudioNavRLEnv", return_value=env), \
                 patch("experiment.avnav.environment.config.get_config", return_value=config), \
                 patch("experiment.avnav.environment.evaluation_episodes.load_episodes",
                       return_value=dataset):
                metrics = evaluate_split("test_multiple", policy, settings, 0, Path(output))

        self.assertEqual(env.steps, [1, 1])
        self.assertEqual(metrics["episodes"], 1)
        self.assertEqual(metrics["errors"], 0)
        policy.act.assert_called()


if __name__ == "__main__":
    unittest.main()
