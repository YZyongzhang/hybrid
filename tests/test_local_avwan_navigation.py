"""Local map recovery and isolation checks; no GPU simulator required."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions as Actions

from experiment import AVWANRLEnv, get_avwan_config
from experiment.avwan.environment.Planner import Planner
from experiment.avwan.environment.nav import LocalEgoMap, LOCAL_SENSOR_TYPES
from soundspaces.tasks.nav import EgoMap


class LocalNavigationTests(unittest.TestCase):
    def setUp(self):
        config = get_avwan_config().TASK_CONFIG.clone()
        config.defrost()
        config.TASK.GEOMETRIC_MAP.INTERNAL_MAP_SIZE = 100
        config.TASK.GEOMETRIC_MAP.MAP_SIZE = 40
        config.freeze()
        self.planner = Planner(task_config=config)
        self.unknown = np.zeros((31, 31, 2), dtype=np.uint8)

    def update(self, ego, collision=False):
        self.planner.update_map_and_graph({
            'ego_map': ego, 'collision': [collision], 'depth': np.zeros((1, 1, 1))})

    def nodes(self, *cells):
        return self.planner._map_index_to_graph_nodes(cells)

    def test_base_graph_is_exact_cardinal_grid(self):
        graph = self.planner._graph
        self.assertEqual(len(graph), 100)
        self.assertEqual(graph.number_of_edges(), 180)
        center = self.nodes((50, 50))[0]
        self.assertEqual(set(graph[center]), set(self.nodes((40, 50), (60, 50), (50, 40), (50, 60))))
        for a, b in graph.edges:
            x1, y1 = graph.nodes[a]['map_index']
            x2, y2 = graph.nodes[b]['map_index']
            self.assertEqual(abs(x2-x1) + abs(y2-y1), 10)

    def test_edge_clears_only_after_two_observed_free_updates(self):
        obstacle = self.unknown.copy()
        obstacle[26, 15] = (1, 1)  # Global (50,45), inside forward edge.
        free = self.unknown.copy()
        free[26, 15] = (0, 1)
        edge = self.nodes((50, 50), (50, 40))
        self.update(obstacle)
        self.assertFalse(self.planner._graph.has_edge(*edge))
        self.update(self.unknown)
        self.assertFalse(self.planner._graph.has_edge(*edge))
        self.update(free)
        self.assertFalse(self.planner._graph.has_edge(*edge))
        self.update(obstacle)  # Occupied evidence resets the free counter.
        self.update(free)
        self.assertFalse(self.planner._graph.has_edge(*edge))
        self.update(free)
        self.assertTrue(self.planner._graph.has_edge(*edge))

    def test_node_and_incident_edges_restore_with_attributes(self):
        obstacle = self.unknown.copy()
        obstacle[21, 15] = (1, 1)  # Global (50,40), a navigation node.
        node = self.nodes((50, 40))[0]
        neighbors = set(self.planner._graph[node])
        self.update(obstacle)
        self.assertNotIn(node, self.planner._graph)
        self.assertFalse(self.planner.check_navigability((50, 40)))
        obstacle[21, 15, 0] = 0
        self.update(obstacle)
        self.update(obstacle)
        self.assertEqual(set(self.planner._graph[node]), neighbors)
        self.assertEqual(self.planner._graph.nodes[node]['map_index'], (50, 40))

    def test_collision_blocks_edge_until_reset_and_replans_around_it(self):
        planner = self.planner
        self.assertEqual(planner.plan({}, (50, 40), False), Actions.MOVE_FORWARD)
        self.update(self.unknown, collision=True)
        edge = self.nodes((50, 50), (50, 40))
        self.assertEqual(planner.mapper.get_maps_and_agent_pose()[2:4], (50, 50))
        self.assertFalse(planner._graph.has_edge(*edge))
        self.assertIn(edge[1], planner._graph)  # Do not discard the free target.
        self.assertEqual(planner._graph.number_of_edges(), planner._base_graph.number_of_edges() - 1)
        self.assertTrue(planner.check_navigability((50, 40)))  # A detour exists.
        self.assertIn(planner.plan({}, (50, 40), False), (Actions.TURN_LEFT, Actions.TURN_RIGHT))
        planner._prev_action = None
        free = self.unknown.copy()
        free[..., 1] = 1
        self.update(free)
        self.update(free)
        self.assertFalse(planner._graph.has_edge(*edge))
        self.assertFalse(planner._graph.has_edge(*reversed(edge)))
        self.assertFalse(planner.mapper._geometric_map[..., 0].any())
        planner.reset()
        self.assertTrue(planner._graph.has_edge(*edge))
        self.assertFalse(planner._collision_edges)
        self.assertFalse(planner.mapper._free_observations.any())

    def test_collision_still_integrates_depth_obstacles(self):
        planner = self.planner
        planner.plan({}, (50, 40), False)
        obstacle = self.unknown.copy()
        obstacle[26, 15] = (1, 1)
        self.update(obstacle, collision=True)
        edge = self.nodes((50, 50), (50, 40))
        self.assertFalse(planner._graph.has_edge(*edge))
        self.assertEqual(planner.mapper.get_maps_and_agent_pose()[2:4], (50, 50))
        obstacle[26, 15, 0] = 0
        planner._prev_action = None
        self.update(obstacle)
        self.update(obstacle)
        self.assertEqual(planner.mapper._geometric_map[45, 50, 0], 0)
        self.assertFalse(planner._graph.has_edge(*edge))

    def test_node_restoration_does_not_restore_collision_edge(self):
        planner = self.planner
        planner.plan({}, (50, 40), False)
        self.update(self.unknown, collision=True)
        planner._prev_action = None
        obstacle = self.unknown.copy()
        obstacle[21, 15] = (1, 1)
        self.update(obstacle)
        target = self.nodes((50, 40))[0]
        self.assertNotIn(target, planner._graph)
        obstacle[21, 15, 0] = 0
        self.update(obstacle)
        self.update(obstacle)
        self.assertIn(target, planner._graph)
        self.assertFalse(planner._graph.has_edge(*self.nodes((50, 50), (50, 40))))
        self.assertEqual(planner._graph.degree(target), planner._base_graph.degree(target) - 1)

    def test_collision_uses_forward_heading_and_ignores_turns(self):
        planner = self.planner
        # A stale next-node hint must not block a different direction.
        planner._prev_next_node = self.nodes((60, 50))[0]
        planner._prev_action = Actions.MOVE_FORWARD
        self.update(self.unknown, collision=True)
        self.assertFalse(planner._graph.has_edge(*self.nodes((50, 50), (50, 40))))
        self.assertTrue(planner._graph.has_edge(*self.nodes((50, 50), (60, 50))))
        blocked = planner._collision_edges.copy()
        planner._prev_action = Actions.TURN_RIGHT
        self.update(self.unknown, collision=True)
        self.assertEqual(planner._collision_edges, blocked)

    def test_collision_disconnection_updates_waypoint_mask(self):
        planner = self.planner
        free = self.unknown.copy()
        free[..., 1] = 1
        for orientation in (0, 90, 180, 270):
            planner.mapper._orientation = orientation
            planner._prev_action = Actions.MOVE_FORWARD
            self.update(free, collision=True)
        self.assertFalse(planner.check_navigability((50, 40)))
        obs = {'action_map': None}
        planner.add_maps_to_observation(obs)
        self.assertEqual(obs['action_map'].sum(), 1)
        self.assertEqual(obs['action_map'][4, 4, 0], 1)

    def test_free_observations_align_across_rotations(self):
        obstacle = self.unknown.copy()
        obstacle[26, 15] = (1, 1)
        self.update(obstacle)
        for action in (Actions.TURN_RIGHT,) * 4:
            self.planner._prev_action = action
            self.update(self.unknown)
        self.planner._prev_action = None
        obstacle[26, 15, 0] = 0
        self.update(obstacle)
        self.update(obstacle)
        self.assertTrue(self.planner._graph.has_edge(*self.nodes((50, 50), (50, 40))))

    def test_mask_includes_only_adjacent_depth_probes_when_disconnected(self):
        planner = self.planner
        current = self.nodes((50, 50))[0]
        paths = []
        for neighbor in planner._graph[current]:
            x, y = planner._graph.nodes[neighbor]['map_index']
            planner.mapper._geometric_map[(50+y)//2, (50+x)//2] = (1, 1)
            paths.append(((50, 50), (x, y)))
        planner._update_graph([], paths)
        obs = {'action_map': None}
        planner.add_maps_to_observation(obs)
        self.assertEqual(obs['action_map'].shape, (9, 9, 1))
        self.assertEqual(obs['action_map'].sum(), 5)  # STOP and four one-edge probes.
        self.assertEqual(obs['action_map'][4, 4, 0], 1)
        self.assertEqual(obs['action_map'][2, 4, 0], 0)  # Cannot probe two edges.
        self.assertEqual(obs['action_map'][3, 3, 0], 0)  # Cannot probe diagonally.
        planner._allow_depth_probes = False
        planner.add_maps_to_observation(obs)
        self.assertEqual(obs['action_map'].sum(), 1)

    def test_successful_depth_probe_verifies_edge_without_erasing_depth(self):
        planner = self.planner
        obstacle = self.unknown.copy()
        obstacle[26, 15] = (1, 1)
        obstacle[21, 15] = (1, 1)  # The endpoint can also have false depth occupancy.
        self.update(obstacle)
        self.assertTrue(planner.is_depth_probe((50, 40)))
        self.assertFalse(planner.check_navigability((50, 40)))
        self.assertTrue(planner.check_waypoint_executability((50, 40)))
        self.assertEqual(planner.plan({}, (50, 40), False), Actions.MOVE_FORWARD)
        self.update(self.unknown, collision=False)
        edge = self.nodes((50, 50), (50, 40))
        self.assertEqual(planner.mapper.get_maps_and_agent_pose()[2:4], (50, 40))
        self.assertTrue(planner._graph.has_edge(*edge))
        self.assertEqual(planner.mapper._geometric_map[45, 50, 0], 1)
        # Even another occupied observation cannot erase executed connectivity.
        planner.mapper._geometric_map[40, 50, 0] = 1
        planner._update_graph([(50, 40)], [((50, 50), (50, 40))])
        self.assertTrue(planner._graph.has_edge(*edge))
        planner.reset()
        self.assertFalse(planner._verified_edges)
        self.assertFalse(planner._verified_nodes)

    def test_failed_depth_probe_cannot_be_retried(self):
        planner = self.planner
        obstacle = self.unknown.copy()
        obstacle[26, 15] = (1, 1)
        obstacle[21, 15] = (1, 1)
        self.update(obstacle)
        self.assertEqual(planner.plan({}, (50, 40), False), Actions.MOVE_FORWARD)
        self.update(self.unknown, collision=True)
        self.assertFalse(planner.is_depth_probe((50, 40)))
        self.assertFalse(planner.check_waypoint_executability((50, 40)))
        obs = {'action_map': None}
        planner.add_maps_to_observation(obs)
        self.assertEqual(obs['action_map'][3, 4, 0], 0)

    def test_expert_probes_when_depth_blocks_all_normal_candidates(self):
        from experiment.avwan.environment.BFSExpert import BFSExpert
        planner = self.planner
        current = self.nodes((50, 50))[0]
        paths = []
        for neighbor in planner._graph[current]:
            x, y = planner._graph.nodes[neighbor]['map_index']
            planner.mapper._geometric_map[(50+y)//2, (50+x)//2] = (1, 1)
            paths.append(((50, 50), (x, y)))
        planner._update_graph([], paths)
        obs = {'action_map': None}
        planner.add_maps_to_observation(obs)
        expert = BFSExpert(get_avwan_config())

        def score(candidates, *args):
            for c in candidates:
                c.distance = 0 if c.goal == (50, 40) else 2
                c.progress = 1 - c.distance
                c.expected_reward = c.progress - 0.01 * c.steps
            return candidates

        with patch.object(expert, '_score_waypoint_candidates', side_effect=score):
            action = expert.select_action(planner=planner, sim=None, observation=obs,
                                          target_positions=[], current_distance=1, goal_reached=False)
        self.assertEqual(action, 31)
        self.assertEqual(expert.last_decision['mode'], 'depth_probe')
        self.assertEqual(expert.last_decision['estimated_steps'], 1)
        self.assertTrue(obs['action_map'].reshape(-1)[action])

    def test_projection_and_masks_near_boundaries(self):
        mapper = self.planner.mapper
        for x, y in ((0, 0), (90, 0), (0, 90), (90, 90)):
            for heading in (0, 90, 180, 270):
                mapper._x, mapper._y, mapper._orientation = x, y, heading
                self.update(np.ones_like(self.unknown))
                self.assertEqual(mapper._geometric_map[y, x, 0], 0)
                mask = mapper.get_egocentric_occupancy_map(9, 1.0)
                self.assertEqual(mask.shape, (9, 9))
                self.assertEqual(mask[4, 4], 1)
        self.assertEqual(self.nodes((-10, 10), (100, 10)), [-1, -1])

    def test_sensor_registration_and_no_shared_cache(self):
        self.assertIs(registry.get_sensor('EgoMap'), EgoMap)
        self.assertIs(registry.get_sensor('LocalEgoMap'), LocalEgoMap)
        config = get_avwan_config().TASK_CONFIG
        sensor = LocalEgoMap(sim=SimpleNamespace(config=config.SIMULATOR),
                             config=config.TASK.EGOMAP_SENSOR)
        shape = sensor.proj_xs.shape
        observation = sensor.get_observation(observations={'depth': np.zeros(shape + (1, 1))}, episode=None)
        self.assertEqual(observation.shape, (31, 31, 2))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertFalse(observation.any())

    def test_projection_does_not_fill_unobserved_gap_with_free_evidence(self):
        config = get_avwan_config().TASK_CONFIG
        sensor = LocalEgoMap(sim=SimpleNamespace(config=config.SIMULATOR),
                             config=config.TASK.EGOMAP_SENSOR)
        # Two floor returns separated by one unobserved cell.
        points = np.array([[-0.2, -sensor.sensor_height, -1.0],
                           [0.0, -sensor.sensor_height, -1.0]])
        with patch.object(sensor, 'convert_to_pointcloud', return_value=points):
            projected = sensor._get_depth_projection(np.zeros((1, 1, 1)))
        self.assertEqual(projected[21, 13, 1], 1)
        self.assertEqual(projected[21, 15, 1], 1)
        self.assertEqual(projected[21, 14, 1], 0)
        self.assertFalse(projected[..., 0].any())

    def test_projection_does_not_invent_obstacles_between_depth_returns(self):
        config = get_avwan_config().TASK_CONFIG
        sensor = LocalEgoMap(sim=SimpleNamespace(config=config.SIMULATOR),
                             config=config.TASK.EGOMAP_SENSOR)

        def point(col, row, height):
            return [(col - 15) * 0.1, height - sensor.sensor_height,
                    (row - 31) * 0.1]

        # Two obstacle strips leave a one-cell-wide gap. Closing used to
        # fabricate occupied evidence in its center (row 21, col 15).
        strips = [point(col, row, 0.8) for col in (14, 16) for row in (20, 21, 22)]
        for name, extra, expected in (
                ('unknown gap', [], (0, 0)),
                ('observed floor', [point(15, 21, 0.0)], (0, 1)),
                ('floor and obstacle', [point(15, 21, 0.0), point(15, 21, 0.8)], (1, 1))):
            with self.subTest(name=name):
                with patch.object(sensor, 'convert_to_pointcloud',
                                  return_value=np.asarray(strips + extra)):
                    projected = sensor._get_depth_projection(np.zeros((1, 1, 1)))
                np.testing.assert_array_equal(projected[21, 15], expected)
                self.assertEqual(projected[21, 14, 0], 1)
                self.assertEqual(projected[21, 16, 0], 1)

    def test_projection_uses_centered_cells_without_shifting_nearby_obstacles(self):
        config = get_avwan_config().TASK_CONFIG
        sensor = LocalEgoMap(sim=SimpleNamespace(config=config.SIMULATOR),
                             config=config.TASK.EGOMAP_SENSOR)
        # A free navigation line at x=1 m, with furniture 8 cm to its side.
        # Floor quantization used to merge both into column 25.
        points = np.array([[1.0, -sensor.sensor_height, -1.8],
                           [1.08, 0.6 - sensor.sensor_height, -1.8],
                           [-1.08, 0.6 - sensor.sensor_height, -1.8]])
        with patch.object(sensor, 'convert_to_pointcloud', return_value=points):
            projected = sensor._get_depth_projection(np.zeros((1, 1, 1)))
        self.assertEqual(projected[13, 25, 0], 0)
        self.assertEqual(projected[13, 25, 1], 1)
        self.assertEqual(projected[13, 26, 0], 1)
        self.assertEqual(projected[13, 4, 0], 1)

    def test_missing_depth_between_obstacles_preserves_known_free_waypoint(self):
        config = get_avwan_config().TASK_CONFIG
        sensor = LocalEgoMap(sim=SimpleNamespace(config=config.SIMULATOR),
                             config=config.TASK.EGOMAP_SENSOR)

        def project(points):
            with patch.object(sensor, 'convert_to_pointcloud', return_value=np.asarray(points)):
                return sensor._get_depth_projection(np.zeros((1, 1, 1)))

        floor = project([[0.0, -sensor.sensor_height, -1.0]])
        self.update(floor)
        node = self.nodes((50, 40))[0]
        self.assertIn(node, self.planner._graph)
        # A later view contains adjacent obstacle returns but no measurement
        # at the waypoint. That missing measurement must not block the node.
        strips = [[x, 0.8 - sensor.sensor_height, z]
                  for x in (-0.1, 0.1) for z in (-1.1, -1.0, -0.9)]
        self.update(project(strips))
        self.assertIn(node, self.planner._graph)
        self.assertTrue(self.planner.check_navigability((50, 40)))
        np.testing.assert_array_equal(self.planner.mapper._geometric_map[40, 50], (0, 1))

    def test_env_selects_local_sensor_without_mutating_caller_config(self):
        config = get_avwan_config()
        sensor_type = config.TASK_CONFIG.TASK.EGOMAP_SENSOR.TYPE
        class StopConstruction(Exception):
            pass
        def inspect_config(instance, task_config, dataset):
            for name, cls in LOCAL_SENSOR_TYPES.items():
                if name in task_config.TASK.SENSORS:
                    self.assertEqual(getattr(task_config.TASK, name).TYPE, cls.__name__)
                    self.assertIs(registry.get_sensor(cls.__name__), cls)
            self.assertTrue(task_config.is_frozen())
            raise StopConstruction()
        with patch('habitat.RLEnv.__init__', inspect_config):
            with self.assertRaises(StopConstruction):
                AVWANRLEnv(config)
        self.assertEqual(config.TASK_CONFIG.TASK.EGOMAP_SENSOR.TYPE, sensor_type)


if __name__ == '__main__':
    unittest.main()
