# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# SOUNDSPACES_LICENSE file in this directory.

# Adapted from SoundSpaces soundspaces/tasks/nav.py (EgoMap only).
# https://github.com/facebookresearch/sound-spaces
# Local registration and depth processing; see README.md.
from typing import Any, Union

import numpy as np
import torch
from gym import spaces
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator


@registry.register_sensor(name="LocalEgoMap")
class LocalEgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_RESOLUTION, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_res = self.config.MAP_RESOLUTION

        # Agent height for pointcloud transformation
        self.sensor_height = self.config.POSITION[1]

        # Compute intrinsic matrix
        hfov = float(self._sim.config.DEPTH_SENSOR.HFOV) * np.pi / 180
        self.intrinsic_matrix = np.array([[1 / np.tan(hfov / 2.), 0., 0., 0.],
                                          [0., 1 / np.tan(hfov / 2.), 0., 0.],
                                          [0., 0.,  1, 0],
                                          [0., 0., 0, 1]])
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.config.DEPTH_SENSOR.WIDTH
        H = self._sim.config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
                                          np.linspace(-1, 1, W),
                                          np.linspace(1, -1, H)
                                     )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 2)
        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.uint8,
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = self.proj_xs.reshape(-1)
        ys = self.proj_ys.reshape(-1)
        depth_float = depth_float.reshape(-1)

        # Filter out invalid depths
        max_forward_range = self.map_size * self.map_res
        valid_depths = np.isfinite(depth_float) & (depth_float > 0.0) & (depth_float <= max_forward_range)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth_float,
                         ys * depth_float,
                         -depth_float, np.ones(depth_float.shape)))
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self._sim.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the point cloud
        XYZ_ego[:, 1] += self.sensor_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2
        points = XYZ_ego

        # Mapper indices denote cell centers at exact metric offsets (and
        # navigation nodes lie on those centers). Floor-only quantization
        # shifts every hit by up to one cell toward negative X/Z, incorrectly
        # putting nearby furniture onto an otherwise free navigation edge.
        grid_x = np.floor((points[:, 0] / self.map_res) + Vby2 + 0.5).astype(int)
        grid_y = np.floor((points[:, 2] / self.map_res) + V + 0.5).astype(int)

        # Filter out invalid points
        valid_idx = (grid_x >= 0) & (grid_x <= V-1) & (grid_y >= 0) & (grid_y <= V-1)
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx]
        grid_y = grid_y[valid_idx]

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)

        # Do not close either channel. Closing can turn a gap with no depth
        # returns into an occupied observation, overwriting known free space
        # in Mapper after a change of viewpoint. Only measured obstacle/floor
        # returns are evidence; missing returns leave the map unchanged.

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        # Keep local projections independent of the simulator's official cache.
        depth = asnumpy(observations['depth'])
        while depth.ndim > 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 2:
            depth = depth[..., None]
        expected = self.proj_xs.shape + (1,)
        if depth.shape != expected:
            raise ValueError(f"Expected depth shape {expected}, got {depth.shape}")
        return self._get_depth_projection(depth).astype(np.uint8)


def asnumpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


# Map placeholders and intensity adapted from ss_baselines/av_wan/avwan_sensors.py.
class MapPlaceHolder(Sensor):
    def __init__(
            self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL),
            dtype=np.uint8,
        )

    def get_observation(
            self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL))


@registry.register_sensor(name="LocalGeometricMap")
class LocalGeometricMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gm"


@registry.register_sensor(name="LocalActionMap")
class LocalActionMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "action_map"


@registry.register_sensor(name="LocalAcousticMap")
class LocalAcousticMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "am"


@registry.register_sensor(name="LocalIntensity")
class LocalIntensity(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "intensity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        num_frame = 150
        audiogoal = self._sim.get_current_audiogoal_observation()
        nonzero_idx = np.min((audiogoal > 0.1 * audiogoal.max()).argmax(axis=1))
        impulse = audiogoal[:, nonzero_idx: nonzero_idx + num_frame]
        rms = np.mean(impulse ** 2)

        return [rms]


LOCAL_SENSOR_TYPES = {
    'EGOMAP_SENSOR': LocalEgoMap,
    'GEOMETRIC_MAP': LocalGeometricMap,
    'ACTION_MAP': LocalActionMap,
    'ACOUSTIC_MAP': LocalAcousticMap,
    'INTENSITY': LocalIntensity,
}
