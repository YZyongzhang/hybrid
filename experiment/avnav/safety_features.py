"""Shared train/evaluation feature definitions for collision-aware AVNav."""

import numpy as np


DEPTH_MAX_M = 10.0
DEPTH_SECTOR_COUNT = 5


def depth_sector_quantiles(observation):
    """Return normalized 20th-percentile depth in five horizontal view sectors."""
    depth = np.asarray(observation["depth"], dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth after squeeze, got {depth.shape}")
    height, width = depth.shape
    depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
    depth = depth[int(height * 0.25):int(height * 0.85)]
    sectors = []
    for index in range(DEPTH_SECTOR_COUNT):
        left, right = round(index * width / DEPTH_SECTOR_COUNT), round(
            (index + 1) * width / DEPTH_SECTOR_COUNT
        )
        sectors.append(float(np.quantile(depth[:, left:right], 0.2)))
    return sectors


def state_progress_delta(distances):
    """Convert a state distance trace into one-step positive-progress features."""
    values = [float(value) for value in distances]
    result = [0.0]
    result.extend(values[index - 1] - values[index]
                  for index in range(1, len(values)))
    return result


def previous_collision_features(collisions):
    """Return collision history aligned to states (reset to zero at episode start)."""
    values = [bool(value) for value in collisions]
    return [0.0] + [float(value) for value in values]
