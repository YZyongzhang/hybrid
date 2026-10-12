import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiment.avwan.collector.collect import collected_episode_keys


class CollectorEpisodeResumeTest(unittest.TestCase):
    def test_only_completed_indexed_episodes_are_skipped(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            saved = root / 'episodes/scene/1.pkl.gz'
            saved.parent.mkdir(parents=True)
            saved.write_bytes(b'complete')
            rows = [
                dict(scene='scene', episode_id='1', status='success', saved=True,
                     file='episodes/scene/1.pkl.gz'),
                dict(scene='scene', episode_id='2', status='success', saved=True,
                     file='episodes/scene/2.pkl.gz'),
                dict(scene='scene', episode_id='3', status='error', saved=False),
                dict(scene='other', episode_id='4', status='failure', saved=False),
            ]
            (root / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(collected_episode_keys(root), {('scene', '1'), ('other', '4')})


if __name__ == '__main__':
    unittest.main()
