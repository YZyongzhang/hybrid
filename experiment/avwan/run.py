"""Run AVWAN training or collection from audiogoal.yaml switches."""
import argparse
import os
from pathlib import Path

from experiment.avwan.config.default import get_config
from experiment.avwan.collector import COLLECTER


def run(config):
    if config.TASK_CONFIG.TRAIN.OPEN:
        from .train import run_training, training_config
        node = config.TASK_CONFIG.TRAIN
        run_training(training_config(node), stage=node.STAGE, resume=node.RESUME)
        return 0
    if not config.TASK_CONFIG.COLLECT.OPEN:
        print('TRAIN.OPEN and COLLECT.OPEN are both false; nothing to run.')
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
    parser.add_argument('--config', help='Optional experiment YAML overrides (TRAIN lives under TASK_CONFIG)')
    args = parser.parse_args(argv)
    previous_directory = Path.cwd()
    try:
        os.chdir(Path(__file__).resolve().parents[2])
        return run(get_config(args.config))
    finally:
        os.chdir(previous_directory)


if __name__ == '__main__':
    raise SystemExit(main())
