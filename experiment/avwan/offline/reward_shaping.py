"""Train-only reward shaping from recorded navigation outcomes."""
import gzip
import math
import pickle
from pathlib import Path

import torch


def resolve_source(manifest, entry):
    roots = manifest['identity']['source']
    roots = roots if isinstance(roots, list) else [roots]
    reference = Path(entry['source'])
    if len(roots) > 1:
        parts = entry['source'].split('/', 1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise ValueError(f"Cannot map cached episode to source: {entry['source']}")
        index, relative = int(parts[0]), parts[1]
        if index >= len(roots):
            raise ValueError(f"Invalid source index in cached episode: {entry['source']}")
        return Path(roots[index]) / relative
    if reference.parts and reference.parts[0].isdigit() and len(reference.parts) > 1:
        reference = Path(*reference.parts[1:])
    return Path(roots[0]) / reference


def apply_episode_reward_shaping(episode, source_path, config):
    """Add progress, long-range success, and episode-failure signals in memory."""
    if not config:
        return {}
    source_path = Path(source_path)
    with gzip.open(source_path, 'rb') as handle:
        raw = pickle.load(handle)
    transitions = raw.get('transitions', [])
    metadata = raw.get('metadata', {})
    if len(transitions) != len(episode['action']) or not transitions:
        raise ValueError(f'Raw/cache transition mismatch: {source_path}')
    if not transitions[-1].get('done'):
        raise ValueError(f'Cannot shape an unfinished episode: {source_path}')

    rewards = episode['reward']
    stats = {'episodes': 1, 'progress_transitions': 0,
             'long_successes': 0, 'failures': 0, 'progress_reward': 0.0,
             'away_penalty': 0.0, 'terminal_adjustment': 0.0}
    success = bool(metadata.get('success', False))
    initial_distance = metadata.get('initial_distance')
    if initial_distance is None or not math.isfinite(float(initial_distance)):
        raise ValueError(f'Missing finite training initial distance: {source_path}')
    initial_distance = float(initial_distance)
    threshold = float(config['long_distance_threshold'])
    if success and initial_distance >= threshold:
        bonus = float(config['long_success_bonus'])
        rewards[-1] += bonus
        stats['long_successes'] = 1
        stats['terminal_adjustment'] += bonus
    if not success:
        penalty = float(config['failure_penalty'])
        rewards[-1] -= penalty
        stats['failures'] = 1
        stats['terminal_adjustment'] -= penalty

    progress_scale = float(config['progress_reward_per_meter'])
    away_scale = float(config['away_penalty_per_meter'])
    previous = initial_distance
    for index, transition in enumerate(transitions):
        distance = transition.get('distance_to_goal')
        if distance is None:
            # Older recovery and correction collectors did not retain this
            # measure. They still receive the episode outcome shaping above.
            continue
        distance = float(distance)
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f'Invalid recorded goal distance: {source_path} transition {index}')
        delta = previous - distance
        if delta >= 0:
            adjustment = progress_scale * delta
            stats['progress_reward'] += adjustment
        else:
            adjustment = away_scale * delta
            stats['away_penalty'] += -adjustment
        rewards[index] += adjustment
        stats['progress_transitions'] += 1
        previous = distance
    return stats


def shape_training_pools(pools, manifest, config):
    """Apply reward shaping only to episodes placed in training pools."""
    totals = {'episodes': 0, 'progress_transitions': 0, 'long_successes': 0,
              'failures': 0, 'progress_reward': 0.0, 'away_penalty': 0.0,
              'terminal_adjustment': 0.0}
    for pool in pools:
        for episode in pool:
            source_path = episode.get('_source_path')
            if source_path is None:
                raise ValueError('Training episode is missing its raw source path')
            stats = apply_episode_reward_shaping(episode, source_path, config)
            for key, value in stats.items():
                totals[key] += value
    return totals
