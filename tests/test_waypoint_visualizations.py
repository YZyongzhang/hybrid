"""Coordinate/snapshot checks without starting a GPU simulator."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiment import AVWANRLEnv
from utils.visualizations import plot_top_down_map_waypoint


class WaypointVisualizationTest(unittest.TestCase):
    def test_map_rotation_empty_history_and_snapshot(self):
        for shape in ((120, 180), (180, 120)):
            for waypoints in ([], [[70, 60], [70, 60]]):
                with self.subTest(shape=shape, waypoints=waypoints):
                    info = {"top_down_map": {
                        "map": np.ones(shape, dtype=np.uint8),
                        "fog_of_war_mask": None,
                        "agent_map_coord": [100, 100], "agent_angle": 0.0,
                        "start_map_coord": [30, 30],
                        "sound_source_map_coords": [[90, 90]],
                        "waypoint_map_coords": waypoints,
                    }}
                    original = deepcopy(info)
                    image = plot_top_down_map_waypoint(info)
                    self.assertEqual(image.shape, (120, 180, 3))
                    self.assertEqual(image.dtype, np.uint8)
                    if waypoints:
                        row, col = ((70, 60) if shape[0] < shape[1]
                                    else (shape[1] - 1 - 60, 70))
                        np.testing.assert_array_equal(image[row, col], [255, 0, 0])
                    np.testing.assert_array_equal(info["top_down_map"]["map"],
                                                  original["top_down_map"]["map"])
                    self.assertEqual(info["top_down_map"]["waypoint_map_coords"], waypoints)

    def test_world_coordinates_use_full_map_and_crop_offset(self):
        env = AVWANRLEnv.__new__(AVWANRLEnv)
        env.start_position = [40.0, 0.0, 30.0]
        env.sound_source_positions = [[90.0, 0.0, 80.0]]
        env.waypoint_positions = [[60.0, 0.0, 70.0]]
        metric = {"top_down_map": {
            "map": np.ones((90, 100), dtype=np.uint8),
            "agent_map_coord": [35, 43],
        }}
        measure = SimpleNamespace(_top_down_map=np.ones((120, 180), dtype=np.uint8))
        env._env = SimpleNamespace(
            sim=SimpleNamespace(get_agent_state=lambda: SimpleNamespace(position=[50, 0, 40])),
            task=SimpleNamespace(measurements=SimpleNamespace(measures={"top_down_map": measure})),
            get_metrics=lambda: metric,
        )

        def to_grid(z, x, shape, sim):
            self.assertEqual(shape, (120, 180))
            return int(z), int(x)

        with patch("experiment.avwan.environment.avwan_env.maps.to_grid", side_effect=to_grid):
            info = env.get_info(None)
        self.assertEqual(info["top_down_map"]["start_map_coord"], [25, 33])
        self.assertEqual(info["top_down_map"]["sound_source_map_coords"], [[75, 83]])
        self.assertEqual(info["top_down_map"]["waypoint_map_coords"], [[65, 53]])
        env.waypoint_positions[0][0] = -1
        self.assertEqual(info["waypoint_positions"], [[60.0, 0.0, 70.0]])
        self.assertNotIn("waypoint_map_coords", metric["top_down_map"])


if __name__ == "__main__":
    unittest.main()
