"""Small integration check; random waypoint transitions are not expert demonstrations."""
import argparse
from pathlib import Path
import random

import numpy as np

from experiment.avwan.config import get_config
from experiment.avwan.environment.avwan_env import AVWANRLEnv
from experiment.avwan.offline import AVWANOfflineDataset, save_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soundspaces-root', default='/home/getuanhui/project/sound-spaces')
    parser.add_argument('--split', default='val_telephone')
    parser.add_argument('--source-sound-dir')
    parser.add_argument('--rir-dir')
    parser.add_argument('--visual-only', action='store_true',
                        help='Test real visual navigation/planning without acoustic sensors; not an audio benchmark')
    parser.add_argument('--output', default='outputs/avwan_smoke/episode.pt')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    random.seed(42)
    np.random.seed(42)
    cfg = get_config(args.soundspaces_root, split=args.split)
    cfg.defrost()
    if args.source_sound_dir:
        cfg.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = str(Path(args.source_sound_dir).resolve())
    if args.rir_dir:
        cfg.TASK_CONFIG.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = str(Path(args.rir_dir).resolve())
    if args.visual_only:
        cfg.TASK_CONFIG.TASK.SENSORS = ['EGOMAP_SENSOR', 'GEOMETRIC_MAP', 'ACTION_MAP', 'COLLISION']
    cfg.TASK_CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL']
    cfg.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 12
    cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
    cfg.freeze()
    with AVWANRLEnv(cfg) as env:
        observation = env.reset()
        print('reset', env.get_current_episode_id(),
              {k: np.shape(v) for k, v in observation.items()}, flush=True)
        transitions = []
        while True:
            candidates = np.flatnonzero(env.action_mask())
            candidates = candidates[candidates != env.stop_action]
            action = int(np.random.choice(candidates)) if len(transitions) < 3 and len(candidates) else env.stop_action
            t = env.step_transition(action)
            transitions.append(t)
            print('step', t['action'], 'duration', t['num_steps'], 'return', t['reward'],
                  'terminated', t['terminated'], 'truncated', t['truncated'], flush=True)
            if t['done']:
                break
        save_transitions(transitions, args.output)
        print('offline samples', len(AVWANOfflineDataset(args.output)), flush=True)
        old_episode = env.get_current_episode_id()
        env.reset()
        print('explicit reset', old_episode, '->', env.get_current_episode_id(), flush=True)
    if args.visual_only:
        print('Visual/planner smoke check only: acoustic observations were disabled.', flush=True)


if __name__ == '__main__':
    main()
