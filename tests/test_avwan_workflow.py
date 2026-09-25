"""Original SoundSpaces sampling and collection of the supplied episodes."""
import contextlib
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiment.avwan.config.default import _C, _TC
from experiment.avwan.collector import COLLECTER, CollectRegister
from experiment.avwan.collector.collect import GreedyCollect, collected_scenes
from experiment.avwan.run import run


def test_config():
    config = _C.clone()
    config.TASK_CONFIG = _TC.clone()
    return config


class WorkflowTests(unittest.TestCase):
    def test_runner_uses_original_per_scene_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'content').mkdir()
            with gzip.open(root / 'train.json.gz', 'wt') as handle:
                json.dump({'episodes': []}, handle)
            for scene, count in [('a', 205), ('b', 103), ('small', 3), ('empty', 0)]:
                episodes = [dict(episode_id=str(i), scene_id=f'{scene}/{scene}.glb',
                                 start_position=[0., 0., 0.], start_rotation=[0., 0., 0., 1.],
                                 goals=[dict(position=[1., 0., 0.])], info={'sound': 'telephone'})
                            for i in range(count)]
                with gzip.open(root / 'content' / f'{scene}.json.gz', 'wt') as handle:
                    json.dump({'episodes': episodes}, handle)
            config = test_config()
            config.TASK_CONFIG.DATASET.DATA_PATH = str(root / 'train.json.gz')
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = ['*']
            config.TASK_CONFIG.DATASET.SCENES_DIR = 'data/scene_datasets/mp3d'
            with patch('experiment.AVWANRLEnv') as env, \
                 patch('experiment.avwan.run.COLLECTER'), \
                 contextlib.redirect_stdout(io.StringIO()):
                run(config)
            dataset = env.call_args.args[1]
            self.assertEqual(len(dataset.episodes), 203)
            self.assertEqual(len({(ep.scene_id, ep.episode_id) for ep in dataset.episodes}), 203)
            for scene, count in [('a', 100), ('b', 100), ('small', 3), ('empty', 0)]:
                self.assertEqual(sum(Path(ep.scene_id).stem == scene for ep in dataset.episodes), count)

    def test_every_env_episode_is_attempted_once_and_resume_skips_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            config = test_config()
            cfg = config.TASK_CONFIG.COLLECT
            cfg.TO_PATH = directory
            cfg.NUM_EPISODES = 100  # Old YAML must no longer truncate collection.
            cfg.VISUALIZE = False
            episodes = [SimpleNamespace(scene_id='scene/scene.glb', episode_id=str(i), info={})
                        for i in range(205)]
            env = SimpleNamespace(habitat_env=SimpleNamespace(episodes=episodes),
                                  seed=lambda seed: None, gamma=.99, action_space=SimpleNamespace(n=81),
                                  stop_action=40, start_position=[0, 0, 0], sound_source_positions=[[1, 0, 0]])
            visited = []

            def collect_episode(current_env):
                eid = current_env.habitat_env.current_episode.episode_id
                visited.append(eid)
                if eid == '101':
                    raise RuntimeError('one episode failed')
                return ([{'action': 40}], dict(status='success', success=True,
                        transitions=1, primitive_steps=1, final_distance=0.), {})

            with patch('experiment.avwan.collector.collect.collect_episode', side_effect=collect_episode), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(COLLECTER(config, env).collect(), 1)
            self.assertEqual(visited, [str(i) for i in range(205)])
            output = Path(directory)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['planned'], 205)
            self.assertEqual(manifest['completed'], 205)
            self.assertEqual(manifest['saved'], 204)
            self.assertEqual(len((output / 'episodes.jsonl').read_text().splitlines()), 205)
            files = list((output / 'episodes').rglob('*.pkl.gz'))
            self.assertEqual(len(files), 204)
            self.assertTrue((output / 'episodes' / 'scene' / '0.pkl.gz').is_file())
            rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['file'], 'episodes/scene/0.pkl.gz')
            before = {p: p.stat().st_mtime_ns for p in files}
            old_index = (output / 'episodes.jsonl').read_bytes()
            old_manifest = (output / 'manifest.json').read_bytes()
            cfg.RESUME = True
            visited.clear()
            with patch('experiment.avwan.collector.collect.collect_episode', side_effect=collect_episode), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(COLLECTER(config, env).collect(), 0)
            self.assertEqual(visited, [])
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in files})
            self.assertEqual(old_index, (output / 'episodes.jsonl').read_bytes())
            self.assertEqual(old_manifest, (output / 'manifest.json').read_bytes())

    def test_scene_resume_keeps_history_and_collects_every_new_scene_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'episodes' / 'old').mkdir(parents=True)
            old_file = output / 'episodes' / 'old' / '42.pkl.gz'
            old_file.write_bytes(b'previously saved episode')
            old_run = '{"dataset_signature": "old full dataset", "source_hashes": {"removed.py": "old"}}'
            (output / 'run.json').write_text(old_run)
            old_row = dict(scene='old', episode_id='42', status='success', saved=True,
                           file='episodes/old/42.pkl.gz', transitions=4)
            old_index = json.dumps(old_row) + '\n'
            (output / 'episodes.jsonl').write_text(old_index)
            (output / 'manifest.json').write_text(json.dumps(dict(
                completed=1, saved=1, success=1, errors=0, transitions=4, planned=1991689)))
            config = test_config()
            cfg = config.TASK_CONFIG.COLLECT
            cfg.TO_PATH, cfg.RESUME, cfg.VISUALIZE = directory, True, False
            # The newly sampled IDs differ from the old run, and scenes interleave.
            episodes = [SimpleNamespace(scene_id=f'{s}/{s}.glb', episode_id=str(i), info={})
                        for s, i in [('old', 99), ('new', 0), ('other', 0), ('new', 1)]]
            env = SimpleNamespace(habitat_env=SimpleNamespace(episodes=episodes),
                                  seed=lambda seed: None, gamma=.99, action_space=SimpleNamespace(n=81),
                                  stop_action=40, start_position=[0, 0, 0], sound_source_positions=[[1, 0, 0]])
            visited = []

            def collect(current_env):
                visited.append(GreedyCollect._key(current_env.habitat_env.current_episode))
                return ([{'action': 40}], dict(status='success', success=True,
                        transitions=1, primitive_steps=1, final_distance=0.), {})

            with patch('experiment.avwan.collector.collect.collect_episode', side_effect=collect), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(COLLECTER(config, env).collect(), 0)
            self.assertEqual(visited, [('new', '0'), ('other', '0'), ('new', '1')])
            self.assertEqual(old_file.read_bytes(), b'previously saved episode')
            self.assertEqual((output / 'run.json').read_text(), old_run)
            index = (output / 'episodes.jsonl').read_text()
            self.assertTrue(index.startswith(old_index))
            self.assertEqual(len(index.splitlines()), 4)
            summary = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(summary['skipped_scenes'], ['old'])
            self.assertEqual(summary['skipped_episodes'], 1)
            self.assertEqual(summary['completed'], 4)
            self.assertEqual(summary['saved'], 4)
            self.assertEqual(summary['transitions'], 7)
            self.assertEqual(summary['planned'], 4)
            self.assertEqual(len(list((output / 'resume_runs').glob('*.json'))), 1)
            cfg.RESUME = False
            with self.assertRaises(FileExistsError):
                COLLECTER(config, env).collect()

    def test_scene_detection_ignores_unfinished_writes_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for scene in ('saved', 'empty', 'interrupted'):
                (output / 'episodes' / scene).mkdir(parents=True)
            (output / 'episodes' / 'saved' / '1.pkl.gz').write_bytes(b'saved')
            (output / 'episodes' / 'interrupted' / '1.pkl.gz.tmp').write_bytes(b'partial')
            (output / 'episodes' / 'legacy_1.pkl.gz').write_bytes(b'legacy')
            (output / 'logs').mkdir()
            for row in [dict(scene='error', status='error', saved=False),
                        dict(scene='filtered', status='timeout', saved=False),
                        dict(scene='missing', status='success', saved=True, file='episodes/missing/1.pkl.gz'),
                        dict(scene='legacy', status='success', saved=True, file='episodes/legacy_1.pkl.gz')]:
                (output / 'logs' / f"{row['scene']}_1.json").write_text(json.dumps(row))
            self.assertEqual(collected_scenes(output), {'saved', 'filtered', 'legacy'})

    def test_factory_uses_supplied_env(self):
        config = test_config()
        env = object()
        collector = COLLECTER(config, env)
        self.assertIsInstance(collector.collecter, GreedyCollect)
        self.assertIs(collector.collecter.env, env)
        self.assertIs(CollectRegister.get('greedy'), CollectRegister.get('avwan_greedy'))
        config.TASK_CONFIG.COLLECT.TYPE = 'missing'
        with self.assertRaises(KeyError):
            COLLECTER(config, env)

    def test_runner_passes_dataset_to_env_and_closes_it(self):
        config = test_config()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = 100
        original_config = config.dump()
        with patch('soundspaces.datasets.audionav_dataset.AudioNavDataset') as dataset, \
             patch('experiment.AVWANRLEnv') as env, \
             patch('experiment.avwan.run.COLLECTER') as collector:
            dataset.return_value.episodes = [object()]
            collector.return_value.collect.return_value = 0
            self.assertEqual(run(config), 0)
            used_config = env.call_args.args[0]
            self.assertEqual(used_config.dump(), original_config)
            self.assertEqual(config.dump(), original_config)
            dataset.assert_called_once_with(config.TASK_CONFIG.DATASET)
            self.assertIs(env.call_args.args[1], dataset.return_value)
            collector.assert_called_once_with(used_config, env.return_value)
            env.return_value.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
