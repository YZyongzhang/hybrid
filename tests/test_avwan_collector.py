"""Data boundaries and SMDP semantics without launching a simulator."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiment.avwan.collector.collect import collect_episode, load_episode, save_episode
from experiment.avwan.environment.BFSExpert import NoReachableWaypointError


class FakeEnv:
    def __init__(self, outcome):
        self.outcome = outcome
        self.steps = 0
        self.reset_count = 0
        self.observation = {'gm': np.zeros((2, 2)), 'spectrogram': np.zeros((2, 2, 2))}

    def reset(self):
        self.reset_count += 1
        return self.observation

    def get_info(self, observation):
        return dict(distance_to_goal=2., spl=0.)

    def get_expert_map_action(self):
        if self.steps and self.outcome == 'blocked':
            raise NoReachableWaypointError('no reachable waypoint')
        return 40 if self.outcome == 'success' else 41

    def step(self, action):
        self.steps += 1
        previous = deepcopy(self.observation)
        self.observation['gm'][:] = 1
        terminated = self.outcome == 'success'
        truncated = self.outcome == 'timeout'
        self.transition = dict(observation=previous, next_observation=deepcopy(self.observation),
            action=action, action_mask=np.ones(81, dtype=bool), num_steps=2,
            reward=1. + .9 * 2., undiscounted_reward=3., discount=0. if terminated else .9**2,
            gamma=.9, low_level_rewards=[1., 2.], low_level_actions=[2, 1],
            terminated=terminated, truncated=truncated, done=terminated or truncated)
        info = dict(distance_to_goal=0. if terminated else 1., spl=1. if terminated else 0.,
                    success=terminated, truncated=truncated, reaching_waypoint=True,
                    cant_reach_waypoint=False, expert_decision={'action': action})
        return self.observation, 3., terminated or truncated, info

    @property
    def last_transition(self):
        return deepcopy(self.transition)


class CollectorTests(unittest.TestCase):
    def test_terminal_observation_reward_and_stop_are_saved_before_reset(self):
        env = FakeEnv('success')
        transitions, metadata, _ = collect_episode(env)
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(metadata['status'], 'success')
        t = transitions[0]
        self.assertEqual(t['action'], 40)
        self.assertEqual(t['reward'], 2.8)
        self.assertEqual(t['discount'], 0.)
        self.assertTrue(t['terminated'])
        self.assertFalse(t['truncated'])
        self.assertTrue((t['observation']['gm'] == 0).all())
        self.assertTrue((t['next_observation']['gm'] == 1).all())
        env.observation['gm'][:] = 7
        self.assertTrue((t['next_observation']['gm'] == 1).all())

    def test_time_limit_preserves_bootstrap(self):
        transitions, metadata, _ = collect_episode(FakeEnv('timeout'))
        self.assertEqual(metadata['status'], 'timeout')
        self.assertTrue(transitions[-1]['truncated'])
        self.assertFalse(transitions[-1]['terminated'])
        self.assertAlmostEqual(transitions[-1]['discount'], .81)

    def test_blocked_prefix_has_explicit_collection_boundary(self):
        transitions, metadata, _ = collect_episode(FakeEnv('blocked'))
        self.assertEqual(metadata['status'], 'blocked')
        self.assertEqual(len(transitions), 1)
        self.assertTrue(transitions[-1]['collector_truncated'])
        self.assertTrue(transitions[-1]['done'])
        self.assertFalse(transitions[-1]['terminated'])
        self.assertAlmostEqual(transitions[-1]['discount'], .81)

    def test_compressed_roundtrip_preserves_masks_and_arrays(self):
        transitions, metadata, _ = collect_episode(FakeEnv('success'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.pkl.gz'
            save_episode(path, transitions, metadata)
            data = load_episode(path)
            self.assertEqual(data['format_version'], 1)
            self.assertEqual(data['metadata'], metadata)
            np.testing.assert_array_equal(data['transitions'][0]['action_mask'], transitions[0]['action_mask'])
            np.testing.assert_array_equal(data['transitions'][0]['next_observation']['gm'],
                                          transitions[0]['next_observation']['gm'])
            self.assertFalse(list(Path(directory).glob('*.tmp')))



if __name__ == '__main__':
    unittest.main()
