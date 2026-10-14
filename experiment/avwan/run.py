"""Run AVWAN testing, foundation training, offline RL, or collection from audiogoal.yaml."""
import argparse
import os
from pathlib import Path

from experiment.avwan.config.default import get_config
from experiment.avwan.collector import COLLECTER


def run(config):
    if config.TASK_CONFIG.TEST.OPEN:
        from .train_offline_rl_reproduction import run_module
        test = config.TASK_CONFIG.TEST
        run_module(
            "experiment.avwan.reproduce_heard_unheard", "evaluate",
            "--output", test.OUTPUT, "--checkpoint", test.CHECKPOINT,
            "--foundation", test.FOUNDATION,
            "--device", test.DEVICE, "--splits", *test.SPLITS,
            "--shards", test.SHARDS,
            "--episodes-per-process", test.EPISODES_PER_PROCESS,
            "--eval-seed", test.EVAL_SEED,
            "--prediction-interval", test.PREDICTION_INTERVAL,
            "--max-rss-mb", test.MAX_RSS_MB,
        )
        return 0
    if config.TASK_CONFIG.TRAIN.OPEN:
        node = config.TASK_CONFIG.TRAIN
        mode = str(node.MODE).lower()
        if mode == 'offline_rl':
            from .train_offline_rl_reproduction import run_from_node
            run_from_node(node.OFFLINE_RL)
        elif mode == 'foundation':
            from .train import run_training, training_config
            foundation = node.FOUNDATION
            run_training(training_config(foundation), stage=foundation.STAGE,
                         resume=foundation.RESUME)
        elif mode == 'foundation_two_stage':
            from .train_offline_rl_reproduction import run_module
            foundation = node.TWO_STAGE_FOUNDATION
            arguments = [
                '--output', foundation.OUTPUT,
                '--train-source', foundation.TRAIN_SOURCE,
                '--heard-validation-source', foundation.HEARD_VALIDATION_SOURCE,
                '--audio-epochs', foundation.AUDIO_EPOCHS,
                '--action-epochs', foundation.ACTION_EPOCHS,
                '--batch-size', foundation.BATCH_SIZE,
                '--workers', foundation.WORKERS,
                '--audio-learning-rate', foundation.AUDIO_LEARNING_RATE,
                '--action-learning-rate', foundation.ACTION_LEARNING_RATE,
                '--grl-scale', foundation.GRL_SCALE,
                '--domain-loss-weight', foundation.DOMAIN_LOSS_WEIGHT,
                '--seed', foundation.SEED,
                '--device', foundation.DEVICE,
                '--gpu-memory-fraction', foundation.GPU_MEMORY_FRACTION,
                '--cpu-threads', foundation.CPU_THREADS,
                '--max-rss-mib', foundation.MAX_RSS_MIB,
                '--min-available-gib', foundation.MIN_AVAILABLE_GIB,
            ]
            unheard_source = foundation.get('UNHEARD_VALIDATION_SOURCE', '')
            if unheard_source:
                arguments.extend(['--unheard-validation-source',
                                  unheard_source])
            run_module('experiment.avwan.train_staged_grl_foundation', *arguments)
        else:
            raise ValueError(
                f'Unsupported TRAIN.MODE: {node.MODE!r}; use foundation, '
                'foundation_two_stage, or offline_rl')
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
