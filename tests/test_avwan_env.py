"""Test the real AVWANRLEnv control flow with deterministic simulator/planner doubles."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment.avwan.offline import AVWANOfflineDataset, load_transitions, save_transitions


def config():
    return NS(CONTINUOUS=False, PREDICTION_INTERVAL=3, MASKING=True, MODEL_DIR=None,
              OFFLINE_RL=NS(GAMMA=0.5),
              RL=NS(WITH_TIME_PENALTY=True, SLACK_REWARD=-0.01,
                    WITH_DISTANCE_REWARD=True, DISTANCE_REWARD_SCALE=1.0, SUCCESS_REWARD=10.),
              TASK_CONFIG=NS(TASK=NS(SUCCESS=NS(SUCCESS_DISTANCE=1.), ACTION_MAP=NS(MAP_SIZE=3),
                                     SENSORS=['EGOMAP_SENSOR', 'COLLISION', 'ACTION_MAP'])))


class CoreEnv:
    def __init__(self, config, dataset=None):
        self.observation_space = None
        self.action_space = None
        self.number_of_episodes = 3
        self.reset_count = 0
        self.events = []
        self.sim = NS(reaching_goal=False)
        self.obs = dict(rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                        depth=np.zeros((2, 2, 1), dtype=np.float32),
                        action_map=np.ones((3, 3, 1)), gm=np.zeros((2, 2, 2)),
                        pointgoal_with_gps_compass=np.array([123., 456.]), pose=np.zeros(2))

    def reset(self):
        self.reset_count += 1
        self.current_episode = NS(episode_id=str(self.reset_count), scene_id='scene',
                                  info={'geodesic_distance':999.})
        self.distance = 10.
        self.task = NS(is_stop_called=False, is_episode_active=True)
        self.episode_over = False
        self.sim.reaching_goal = False
        self.steps = 0
        self.obs['rgb'].fill(0)
        self.obs['pose'].fill(0)
        return self.obs

    def step(self, action):
        event = self.events.pop(0) if self.events else {}
        self.steps += 1
        self.distance = event.get('distance', self.distance - 1)
        self.task.is_stop_called = action['action'] == 0
        self.task.is_episode_active = not (self.task.is_stop_called or event.get('task_end', False))
        self.episode_over = not self.task.is_episode_active or event.get('timeout', False)
        self.sim.reaching_goal = self.distance == 0
        self.obs['rgb'].fill(self.steps)
        self.obs['pose'][:] = event.get('pose', (0, 0))
        return self.obs

    def get_metrics(self):
        return {'distance_to_goal':self.distance, 'spl':float(self.sim.reaching_goal)}


class FakePlanner:
    def __init__(self, **kwargs):
        self.mapper = self
        self.reset_count = 0
        self.navigable = True
        self.reset()

    def reset(self):
        self.reset_count += 1
        self.x = self.y = 0
        self.updates = 0

    def update_map_and_graph(self, observation):
        self.updates += 1
        self.x, self.y = observation['pose']

    def add_maps_to_observation(self, observation):
        observation['gm'].fill(self.updates)

    def get_map_coordinates(self, action):
        return (action % 3 - 1, action // 3 - 1)

    def check_navigability(self, goal):
        return self.navigable

    def plan(self, observation, goal, stop):
        return 0 if stop else 1

    def get_maps_and_agent_pose(self):
        return None, None, self.x, self.y, 0


@unittest.skipUnless(importlib.util.find_spec('habitat'), 'Habitat is needed for environment contract tests')
class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from experiment.avwan.environment.avwan_env import AVWANRLEnv
        cls.Env = AVWANRLEnv

    def setUp(self):
        self.addCleanup(patch.stopall)
        patch('habitat.core.env.Env', CoreEnv).start()
        patch('experiment.avwan.environment.avwan_env.Planner', FakePlanner).start()
        self.env = self.Env(config())
        self.env.reset()

    def test_waypoint_duration_rewards_and_owned_observations(self):
        obs, reward, done, info = self.env.step(action=8)
        t = self.env.last_transition
        self.assertFalse(done)
        self.assertAlmostEqual(reward, 2.97)
        self.assertEqual(t['num_steps'], 3)
        self.assertAlmostEqual(t['reward'], .99 * (1 + .5 + .25))
        self.assertAlmostEqual(t['discount'], .125)
        self.assertEqual(info['low_level_actions'], [1, 1, 1])
        self.assertEqual(self.env.action_space.n, 9)
        self.assertEqual(t['observation']['rgb'].max(), 0)
        self.assertEqual(t['next_observation']['rgb'].max(), 3)
        obs['rgb'].fill(99)
        self.env.step(8)
        self.assertEqual(t['next_observation']['rgb'].max(), 3)
        self.assertEqual(self.env.last_transition['observation']['rgb'].max(), 3)

    def test_terminal_stop_keeps_final_observation_and_requires_reset(self):
        self.env.habitat_env.events = [{'distance':0}]
        obs, reward, done, info = self.env.step({'action':4})
        self.assertTrue(done)
        self.assertTrue(info['success'])
        self.assertTrue(info['terminated'])
        self.assertFalse(info['truncated'])
        self.assertFalse(info['reaching_waypoint'])
        self.assertEqual(self.env.habitat_env.reset_count, 1)
        self.assertEqual(obs['rgb'].max(), 1)
        self.assertEqual(obs['gm'].max(), 2)  # Terminal map includes final observation.
        self.assertEqual(self.env.last_transition['discount'], 0)
        self.assertAlmostEqual(reward, 19.99)
        with self.assertRaises(RuntimeError):
            self.env.step(8)
        prior = self.env.last_transition
        self.env.reset()
        self.assertEqual(self.env.get_current_episode_id(), '2')
        self.assertEqual(prior['episode_id'], '1')
        self.assertEqual(self.env.planner.updates, 1)
        with self.assertRaises(RuntimeError):
            _ = self.env.last_transition

    def test_failed_stop_and_timeout_bootstrap_differ(self):
        t = self.env.step_transition(4)
        self.assertTrue(t['terminated'])
        self.assertEqual(t['discount'], 0)
        self.assertLess(t['reward'], 10)
        self.env.reset()
        self.env.habitat_env.events = [{}, {'timeout':True}]
        t = self.env.step_transition(8)
        self.assertTrue(t['truncated'])
        self.assertFalse(t['terminated'])
        self.assertEqual(t['num_steps'], 2)
        self.assertEqual(t['discount'], .25)
        self.assertEqual(self.env.habitat_env.reset_count, 2)

    def test_stop_at_time_limit_is_terminal(self):
        self.env.habitat_env.events = [{'timeout':True}]
        t = self.env.step_transition(4)
        self.assertTrue(t['terminated'])
        self.assertFalse(t['truncated'])

    def test_reaching_waypoint_does_not_end_episode(self):
        self.env.habitat_env.events = [{'pose':(1, 0)}]
        _, _, done, info = self.env.step(5)
        self.assertTrue(info['reaching_waypoint'])
        self.assertEqual(info['num_steps'], 1)
        self.assertFalse(done)
        self.assertEqual(self.env.last_transition['discount'], .5)

    def test_unreachable_goal_executes_fallback_without_zero_step_transition(self):
        self.env.planner.navigable = False
        _, _, done, info = self.env.step(8)
        self.assertFalse(done)
        self.assertTrue(info['cant_reach_waypoint'])
        self.assertEqual(info['num_steps'], 1)

    def test_stationary_collision_like_step_still_costs_time(self):
        self.env.habitat_env.events = [{'distance':10}] * 3
        _, reward, done, info = self.env.step(8)
        self.assertAlmostEqual(reward, -.03)
        self.assertFalse(done)
        self.assertEqual(info['num_steps'], 3)

    def test_mask_invalid_actions_and_continuous_success(self):
        self.env._previous_observation['action_map'].fill(0)
        mask = self.env.action_mask()
        self.assertEqual(np.flatnonzero(mask).tolist(), [4])
        for action in (-1, 9, True, 2.0, [2], {'action':1, 'other':2}):
            with self.assertRaises(ValueError):
                self.env.step(action)
        self.assertEqual(self.env.habitat_env.steps, 0)
        self.env._continuous = True
        self.env.habitat_env.events = [{'distance':.5}]
        _, _, _, info = self.env.step(np.int64(4))
        self.assertTrue(info['success'])

    def test_nonfinite_distance_rejected(self):
        self.env.habitat_env.distance = float('inf')
        with self.assertRaises(ValueError):
            self.env.get_reward({})

    def test_cached_depth_and_multiple_source_audio_shapes(self):
        self.env.habitat_env.obs['depth'] = np.zeros((2, 2, 1, 1))
        self.env.habitat_env.obs['spectrogram'] = (np.ones((65, 26, 2)), np.ones((2, 16000)))
        observation = self.env.reset()
        self.assertEqual(observation['depth'].shape, (2, 2, 1))
        self.assertEqual(observation['spectrogram'].shape, (65, 26, 2))
        t = self.env.step_transition(4)
        self.assertEqual(t['next_observation']['depth'].dtype, np.float32)
        self.assertEqual(t['next_observation']['spectrogram'].dtype, np.float32)

    def test_offline_roundtrip_dataloader_and_schema_validation(self):
        first = self.env.step_transition(8)
        last = self.env.step_transition(4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.pt'
            save_transitions([first, last], path)
            loaded = load_transitions(path)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(torch.equal(loaded[1]['next_observation']['rgb'],
                                        torch.as_tensor(last['next_observation']['rgb'])))
            dataset = AVWANOfflineDataset(folder)
            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(batch['action'].tolist(), [8, 4])
            self.assertEqual(batch['num_steps'].tolist(), [3, 1])
            self.assertNotIn('pointgoal_with_gps_compass', batch['observation'])
            target = batch['reward'] + batch['discount'] * torch.ones(2)
            self.assertEqual(target[1], batch['reward'][1])
            with self.assertRaises(FileExistsError):
                save_transitions([first], path)
            bad = deepcopy(first)
            bad['discount'] = .99
            with self.assertRaises(ValueError):
                save_transitions([bad], Path(folder) / 'bad.pt')
            torch.save({'actions':torch.tensor([1])}, Path(folder) / 'avnav.pt')
            with self.assertRaises(ValueError):
                load_transitions(Path(folder) / 'avnav.pt')

            # Offline workers must not import Habitat/SoundSpaces at all.
            import subprocess
            import sys
            result = subprocess.run([sys.executable, '-c',
                "import sys; sys.modules['habitat'] = None; sys.modules['ss_baselines'] = None; "
                "from experiment.avwan.offline import AVWANOfflineDataset; "
                "assert len(AVWANOfflineDataset(sys.argv[1])) == 2", str(path)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec('habitat'), 'Habitat is needed for the official planner')
class OfficialPlannerTest(unittest.TestCase):
    def test_real_planner_coordinate_frame_and_collision_pose(self):
        from experiment.avwan.config import get_config
        from ss_baselines.av_wan.models.planner import Planner
        from habitat.sims.habitat_simulator.actions import HabitatSimActions
        cfg = get_config()
        cfg.defrost()
        cfg.TASK_CONFIG.TASK.GEOMETRIC_MAP.INTERNAL_MAP_SIZE = 100
        cfg.TASK_CONFIG.TASK.GEOMETRIC_MAP.MAP_SIZE = 20
        cfg.freeze()
        self.assertIn('DEPTH_SENSOR', cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS)
        self.assertTrue(Path(cfg.TASK_CONFIG.DATASET.DATA_PATH).is_absolute())
        planner = Planner(task_config=cfg.TASK_CONFIG, use_acoustic_map=False)
        obs = dict(ego_map=np.zeros((31, 31, 2)), depth=np.ones((128, 128, 1)),
                   collision=np.array([False]), action_map=np.zeros((9, 9, 1)))
        planner.update_map_and_graph(obs)
        planner.add_maps_to_observation(obs)
        x, y = planner.mapper.get_maps_and_agent_pose()[2:4]
        goal = planner.get_map_coordinates(31)  # One cell straight ahead, initial orientation -Z.
        self.assertEqual(goal, (x, y - 10))
        self.assertEqual(planner.plan(obs, goal, stop=False), HabitatSimActions.MOVE_FORWARD)
        obs['collision'][0] = True
        planner.update_map_and_graph(obs)
        self.assertEqual(planner.mapper.get_maps_and_agent_pose()[2:4], (x, y))
        self.assertEqual(planner.get_map_coordinates(40), (x, y))
        self.assertEqual(planner.plan(obs, (x, y), stop=True), HabitatSimActions.STOP)

    def test_real_acoustic_map_updates(self):
        from experiment.avwan.config import get_config
        from ss_baselines.av_wan.models.planner import Planner
        cfg = get_config()
        cfg.defrost()
        cfg.TASK_CONFIG.TASK.GEOMETRIC_MAP.INTERNAL_MAP_SIZE = 100
        cfg.TASK_CONFIG.TASK.GEOMETRIC_MAP.MAP_SIZE = 20
        cfg.freeze()
        planner = Planner(task_config=cfg.TASK_CONFIG, use_acoustic_map=True)
        observation = dict(ego_map=np.zeros((31, 31, 2)), depth=np.ones((128, 128, 1)),
                           collision=np.array([False]), intensity=np.array([.5]),
                           am=np.zeros((20, 20, 1)))
        planner.update_map_and_graph(observation)
        planner.add_maps_to_observation(observation)
        self.assertEqual(observation['am'].shape, (20, 20, 1))
        self.assertEqual(observation['am'].dtype, np.float32)
        self.assertEqual(observation['am'].max(), .5)


if __name__ == '__main__':
    unittest.main()
