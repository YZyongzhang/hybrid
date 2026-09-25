"""Collect BFSExpert waypoint transitions. Run from the repository root."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]



def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_episode(path, transitions, metadata):
    """Atomic, compressed episode file; observations remain owned NumPy arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wb') as handle:
        pickle.dump(dict(format_version=1, metadata=metadata, transitions=transitions),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_episode(path):
    """Read a trusted collector file (pickle is not safe for untrusted input)."""
    with gzip.open(path, 'rb') as handle:
        return pickle.load(handle)


def collected_scenes(output):
    """Snapshot scenes with saved episodes or a completed collection result."""
    output = Path(output)
    scenes = set()
    for directory in (output / 'episodes').glob('*'):
        if directory.is_dir() and next(directory.glob('*.pkl.gz'), None) is not None:
            scenes.add(directory.name)
    # Also covers SUCCESS_ONLY runs and the older flat episode-file layout.
    # An error, empty directory, or unfinished .tmp file is not collected data.
    for path in (output / 'logs').glob('*.json'):
        row = json.loads(path.read_text())
        if row.get('status') in ('success', 'failure', 'timeout', 'blocked'):
            if not row.get('saved') or (output / row['file']).is_file():
                scenes.add(str(row['scene']))
    return scenes


def collect_episode(env):
    """Keep final observations before any reset and use the env's SMDP return."""
    from experiment.avwan.environment.BFSExpert import NoReachableWaypointError
    observation = env.reset()
    info = env.get_info(observation)
    initial_distance = float(info['distance_to_goal'])
    transitions, blocked = [], None
    done = False
    while not done:
        try:
            action = env.get_expert_map_action()
        except NoReachableWaypointError as error:
            blocked = str(error)
            # Collection stopped, not a task terminal. Bootstrap from the last
            # real next_observation; never invent a STOP or a zero-step action.
            if transitions:
                transitions[-1].update(done=True, truncated=True,
                                       collector_truncated=True)
            break
        observation, _, done, info = env.step(action)
        transition = env.last_transition
        if not transition['action_mask'][action]:
            raise ValueError(f'Expert selected masked action {action}')
        transition['expert_decision'] = deepcopy(info.get('expert_decision'))
        transition['reaching_waypoint'] = bool(info['reaching_waypoint'])
        transition['cant_reach_waypoint'] = bool(info['cant_reach_waypoint'])
        transition['distance_to_goal'] = float(info['distance_to_goal'])
        transitions.append(transition)
    success = bool(info.get('success', False))
    status = ('success' if success else 'blocked' if blocked else
              'timeout' if info.get('truncated', False) else 'failure')
    metadata = dict(status=status, success=success, blocked=blocked,
                    initial_distance=initial_distance,
                    final_distance=float(info['distance_to_goal']), spl=float(info['spl']),
                    transitions=len(transitions),
                    primitive_steps=sum(t['num_steps'] for t in transitions))
    return transitions, metadata, info


class GreedyCollect:
    """Collect supplied episodes, optionally skipping previously collected scenes."""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.collect_config = config.TASK_CONFIG.COLLECT

    def collect(self):
        cfg = self.collect_config
        if not cfg.OPEN:
            print('TASK_CONFIG.COLLECT.OPEN is false; collection disabled.')
            return 0
        output_dir = cfg.TO_PATH or cfg.OUTPUT_DIR
        if not output_dir.strip():
            raise ValueError('Collection output directory is required')
        output = Path(output_dir).expanduser().resolve()
        episodes = self.env.habitat_env.episodes
        dataset_count = len(episodes)
        # Explicit episode/selection filters are for debugging. NUM_EPISODES is
        # a legacy field and never truncates the environment's dataset.
        if cfg.EPISODE_ID is not None and str(cfg.EPISODE_ID):
            episodes = [ep for ep in episodes if str(ep.episode_id) == str(cfg.EPISODE_ID)]
            if len(episodes) != 1:
                raise ValueError('EPISODE_ID must identify one episode in the dataset')
        if cfg.SELECTION_PATH:
            requested = json.loads(Path(cfg.SELECTION_PATH).expanduser().read_text())
            keys = {(str(row['scene']), str(row['episode_id'])) for row in requested}
            episodes = [ep for ep in episodes if self._key(ep) in keys]
            if {self._key(ep) for ep in episodes} != keys:
                raise ValueError('Selection contains episodes absent from the environment dataset')
        if not episodes:
            raise ValueError('Environment dataset contains no episodes to collect')
        if not cfg.RESUME and output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output is not empty: {output}; set COLLECT.RESUME or use a new directory')
        # Take the snapshot before collecting. Saving the first episode of a new
        # scene must not cause its remaining episodes to be skipped in this run.
        existing_scenes = collected_scenes(output) if cfg.RESUME else set()
        skipped_scenes = sorted({self._key(ep)[0] for ep in episodes} & existing_scenes)
        pending = [(index, ep) for index, ep in enumerate(episodes)
                   if self._key(ep)[0] not in existing_scenes]
        for scene in skipped_scenes:
            print(f'RESUME: skipping collected scene {scene}', flush=True)
        if not pending:
            print('All selected scenes already have collected data; nothing to collect.', flush=True)
            return 0
        signature = hashlib.sha256()
        for ep in episodes:
            signature.update(json.dumps(self._key(ep)).encode())
            signature.update(b'\n')
        output.mkdir(parents=True, exist_ok=True)
        sources = list((ROOT / 'experiment/avwan').rglob('*.py')) + [ROOT / 'run_avwan.py']
        run_path = output / 'run.json'
        if run_path.exists():
            # Scene resume deliberately allows new random samples and changed
            # collector code. Preserve the old provenance and record this run.
            (output / 'resume_runs').mkdir(exist_ok=True)
            run_path = output / 'resume_runs' / f'{uuid4().hex}.json'
        write_json(run_path, dict(format_version=1, policy='BFSExpert',
            config=self.config.dump(), dataset_count=dataset_count, planned=len(pending),
            skipped_scenes=skipped_scenes, dataset_signature=signature.hexdigest(),
            source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources}))
        for directory in ('episodes', 'logs', 'topdownmaps'):
            (output / directory).mkdir(exist_ok=True)
        summary = dict(format_version=1, dataset_count=dataset_count, planned=len(pending),
                       completed=0, saved=0, success=0, errors=0, transitions=0,
                       episodes_index='episodes.jsonl', skipped_scenes=skipped_scenes,
                       skipped_episodes=len(episodes) - len(pending))
        if cfg.RESUME and (output / 'manifest.json').exists():
            previous = json.loads((output / 'manifest.json').read_text())
            for key in ('completed', 'saved', 'success', 'errors', 'transitions'):
                summary[key] = previous.get(key, 0)
            summary['planned'] += summary['completed']
        write_json(output / 'manifest.json', summary)
        # An incremental index avoids keeping millions of result dictionaries
        # in memory or rewriting the entire history after each episode.
        new_errors = 0
        with (output / 'episodes.jsonl').open('a' if cfg.RESUME else 'w') as index_file:
            for progress, (index, episode) in enumerate(pending, 1):
                scene, episode_id = self._key(episode)
                name = f'{scene}_{episode_id}'
                result_path = output / 'logs' / f'{name}.json'
                row = json.loads(result_path.read_text()) if result_path.exists() else None
                complete = row is not None and row['status'] != 'error'
                if complete and row.get('saved'):
                    complete = (output / row['file']).is_file()
                if not complete:
                    # Habitat's public setter makes reset() use this episode.
                    # This also prevents iterator shuffle, cycling, or a failed
                    # reset from skipping/repeating an episode during collection.
                    self.env.habitat_env.current_episode = episode
                    self.env.seed((int(self.config.SEED) + index) % 2**32)
                    row = dict(scene=scene, episode_id=episode_id, status='error', saved=False)
                    try:
                        transitions, metrics, info = collect_episode(self.env)
                        row.update(metrics, gamma=self.env.gamma,
                                   action_count=self.env.action_space.n, stop_action=self.env.stop_action,
                                   prediction_interval=self.config.PREDICTION_INTERVAL,
                                   start_position=self.env.start_position,
                                   sound_source_positions=self.env.sound_source_positions,
                                   sound=episode.info.get('sound') if episode.info else None)
                        if cfg.VISUALIZE:
                            from PIL import Image
                            from utils.visualizations import plot_top_down_map_waypoint
                            Image.fromarray(plot_top_down_map_waypoint(info)).save(
                                output / 'topdownmaps' / f'{name}.png')
                        if transitions and (not cfg.SUCCESS_ONLY or row['success']):
                            row.update(saved=True, file=f'episodes/{scene}/{episode_id}.pkl.gz')
                            save_episode(output / row['file'], transitions, row)
                    except Exception:
                        row.update(status='error', error=traceback.format_exc())
                    write_json(result_path, row)
                summary['completed'] += 1
                summary['saved'] += int(row.get('saved', False))
                summary['success'] += int(row['status'] == 'success')
                summary['errors'] += int(row['status'] == 'error')
                new_errors += int(row['status'] == 'error')
                if row.get('saved'):
                    summary['transitions'] += row.get('transitions', 0)
                index_file.write(json.dumps(row) + '\n')
                index_file.flush()
                write_json(output / 'manifest.json', summary)
                print(f"{progress}/{len(pending)} {name}: {row['status']}, saved={row.get('saved')}", flush=True)
        return int(new_errors > 0)

    @staticmethod
    def _key(episode):
        scene = Path(episode.scene_id).stem
        episode_id = str(episode.episode_id)
        if Path(episode_id).name != episode_id or episode_id in ('', '.', '..'):
            raise ValueError(f'Invalid episode ID: {episode_id}')
        return scene, episode_id
