"""AVWAN data collection using the existing environment config."""
import argparse
import os
from pathlib import Path

from experiment.avwan.config.default import get_config
from experiment.avwan.collector import COLLECTER


def run(config):
    if not config.TASK_CONFIG.COLLECT.OPEN:
        print('TASK_CONFIG.COLLECT.OPEN is false; collection disabled.')
        return 0
    from experiment import AVWANRLEnv
    from soundspaces.datasets.audionav_dataset import AudioNavDataset

    dataset = AudioNavDataset(config.TASK_CONFIG.DATASET)
    if not dataset.episodes:
        raise ValueError('Configured dataset contains no episodes')
    env = AVWANRLEnv(config, dataset)
    try:
        return COLLECTER(config, env).collect()
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Optional YAML overrides for config/default.py')
    args = parser.parse_args(argv)
    previous_directory = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[2])
        return run(get_config(args.config))
    finally:
        os.chdir(previous_directory)


if __name__ == '__main__':
    raise SystemExit(main())
