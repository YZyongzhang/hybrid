"""Waypoint decisions over the planner's current graph, with episode memory.

This is an oracle for data collection: real goal distances score known nodes,
but graph search uses only the planner's connectivity. It never edits that graph
or executes simulator actions. Unknown targets can be selected for exploration.
"""
from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Optional

import networkx as nx
import numpy as np
from habitat.utils.geometry_utils import quaternion_rotate_vector


class NoReachableWaypointError(RuntimeError):
    """No movement target can be expressed in the current waypoint action map.

    The caller must handle the blockage explicitly. A turn-only action is not
    part of the waypoint action space, and its center means STOP, not scanning.
    """


@dataclass
class _Candidate:
    action: int
    goal: tuple
    steps: int
    distance: Optional[float] = None
    progress: float = 0.0
    expected_reward: float = 0.0
    unexplored: float = 0.0
    penalty: float = 0.0


class BFSExpert:
    """Select progress waypoints, then exploration/detour targets if blocked.

    Search counts turns and forward moves in (planner node, heading) space.
    Costs are estimates: the planner can choose another path or discover new
    obstacles during execution. History is updated only from executed moves.
    """

    def __init__(self, config, *, exploration_weight=1.0, visit_penalty=0.5,
                 failure_penalty=2.0, recent_window=8):
        task = config.TASK_CONFIG.TASK
        self.map_size = int(task.ACTION_MAP.MAP_SIZE)
        self.stop_action = self.map_size ** 2 // 2
        self.resolution = float(task.ACTION_MAP.MAP_RESOLUTION)
        self.stride = int(round(self.resolution / task.GEOMETRIC_MAP.MAP_RESOLUTION))
        self.prediction_interval = int(config.PREDICTION_INTERVAL)
        self.exploration_radius = max(1, int(task.EGOMAP_SENSOR.MAP_SIZE) // 2)
        self.slack = float(config.RL.SLACK_REWARD) if config.RL.WITH_TIME_PENALTY else 0.0
        self.distance_scale = (float(config.RL.DISTANCE_REWARD_SCALE)
                               if config.RL.WITH_DISTANCE_REWARD else 0.0)
        self.exploration_weight = float(exploration_weight)
        self.visit_penalty = float(visit_penalty)
        self.failure_penalty = float(failure_penalty)
        if self.map_size < 3 or self.map_size % 2 != 1 or self.stride < 1:
            raise ValueError('BFSExpert requires an odd action map and a positive map stride')
        if self.prediction_interval < 1 or recent_window < 1:
            raise ValueError('Search horizon and recent history window must be positive')
        if any(not math.isfinite(v) or v < 0 for v in
               (self.exploration_weight, self.visit_penalty, self.failure_penalty)):
            raise ValueError('Expert scoring weights must be finite and nonnegative')
        self._recent_window = recent_window
        self.reset()

    def reset(self, start_cell=None):
        self.visit_counts = Counter()
        self.failed_targets = Counter()
        self.recent_cells = deque(maxlen=self._recent_window)
        self._last_cell = None
        self.last_decision = None
        if start_cell is not None:
            self._record_visit(start_cell)

    def _record_visit(self, cell):
        cell = tuple(int(v) for v in cell)
        # Turning or waiting in place is not another visit to the same cell.
        if cell != self._last_cell:
            self.visit_counts[cell] += 1
            self.recent_cells.append(cell)
            self._last_cell = cell

    def observe_transition(self, *, action, waypoint_goal, visited_cells,
                           reached, blocked):
        """Record actual primitive-step positions, not intended target visits."""
        moved = False
        for cell in visited_cells:
            cell = tuple(int(v) for v in cell)
            moved |= self._last_cell is not None and cell != self._last_cell
            self._record_visit(cell)
        goal = tuple(int(v) for v in waypoint_goal)
        if reached:
            self.failed_targets.pop(goal, None)
        elif action != self.stop_action and (blocked or not moved):
            self.failed_targets[goal] += 1

    def action_to_world_position(self, action, agent_state):
        row, col = divmod(action, self.map_size)
        center = self.map_size // 2
        offset = np.array([(col - center) * self.resolution, 0.0,
                           (row - center) * self.resolution])
        return (np.asarray(agent_state.position, dtype=np.float64)
                + quaternion_rotate_vector(agent_state.rotation, offset)).tolist()

    @staticmethod
    def waypoint_to_nav_node(sim, position, reachable_nodes=None):
        """Match X/Z within 1 mm and height within 0.5 m of the current pose."""
        origin = np.asarray(sim.get_agent_state().position, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        if reachable_nodes is None:
            current_node = sim._position_to_index(origin.tolist())
            reachable_nodes = nx.node_connected_component(sim.graph, current_node)
        nodes = list(reachable_nodes)
        if not nodes:
            return None
        points = np.asarray([sim.graph.nodes[node]['point'] for node in nodes], dtype=np.float64)
        height_difference = np.abs(points[:, 1] - origin[1])
        matches = np.flatnonzero((height_difference <= 0.5)
                                & np.all(np.abs(points[:, [0, 2]] - position[[0, 2]]) <= 1e-3, axis=1))
        if not len(matches):
            return None
        return nodes[matches[np.argmin(height_difference[matches])]]

    def _search_reachable_waypoints(self, planner, observation, horizon):
        mask = np.asarray(observation['action_map']).reshape(-1) > 0.5
        if mask.size != self.map_size ** 2:
            raise ValueError('action_map shape does not match BFSExpert action space')
        _, _, x, y, orientation = planner.mapper.get_maps_and_agent_pose()
        start = planner._map_index_to_graph_nodes([(x, y)])[0]
        graph = planner._graph
        if start not in graph:
            return []
        targets = {}
        for action in range(self.map_size ** 2):
            if action == self.stop_action or not mask[action]:
                continue
            goal = tuple(int(v) for v in planner.get_map_coordinates(action))
            node = planner._map_index_to_graph_nodes([goal])[0]
            if node in graph:
                targets[node] = (action, goal)

        # Planner coordinates: east=0, south=1, west=2, north=3.
        heading = int(round(orientation / 90)) % 4
        queue = deque([(start, heading, 0)])
        visited = {(start, heading)}
        found = set()
        candidates = []
        while queue:
            node, heading, steps = queue.popleft()
            if node in targets and node not in found:
                action, goal = targets[node]
                candidates.append(_Candidate(action, goal, steps))
                found.add(node)
            if steps >= horizon:
                continue
            successors = [(node, (heading - 1) % 4), (node, (heading + 1) % 4)]
            nx_, ny_ = graph.nodes[node]['map_index']
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[heading]
            for neighbor in graph.neighbors(node):
                tx, ty = graph.nodes[neighbor]['map_index']
                if (tx - nx_, ty - ny_) == (dx * self.stride, dy * self.stride):
                    successors.append((neighbor, heading))
            for next_state in successors:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, steps + 1))
        return candidates

    def _score_waypoint_candidates(self, candidates, planner, sim, target_positions, current_distance):
        agent_state = sim.get_agent_state()
        current_node = sim._position_to_index(agent_state.position.tolist())
        reachable = nx.node_connected_component(sim.graph, current_node)
        target_nodes = [sim._position_to_index(list(position)) for position in target_positions]
        distances = nx.multi_source_dijkstra_path_length(sim.graph, target_nodes, weight=None)
        grid_size = float(sim.config.GRID_SIZE)
        geometric_map = planner.mapper.get_maps_and_agent_pose()[0]
        for candidate in candidates:
            world = self.action_to_world_position(candidate.action, agent_state)
            node = self.waypoint_to_nav_node(sim, world, reachable)
            if node in distances:
                candidate.distance = float(distances[node] * grid_size)
                candidate.progress = current_distance - candidate.distance
            # Unknown real nodes remain eligible for exploration, but are not
            # credited with oracle progress that has not been established.
            candidate.expected_reward = (self.distance_scale * candidate.progress
                                         + self.slack * candidate.steps)
            x, y = candidate.goal
            r = self.exploration_radius
            patch = geometric_map[max(0, y-r):y+r+1, max(0, x-r):x+r+1, 1]
            candidate.unexplored = float(np.mean(patch <= 0.5)) if patch.size else 0.0
            candidate.penalty = (self.visit_penalty * self.visit_counts[candidate.goal]
                                 + self.failure_penalty * self.failed_targets[candidate.goal]
                                 + self.visit_penalty * self.recent_cells.count(candidate.goal))
        return candidates

    @staticmethod
    def _select_progress_waypoint(candidates):
        progress = [c for c in candidates if c.distance is not None and c.progress > 1e-6
                    and c.expected_reward - c.penalty > 0]
        return max(progress, key=lambda c: (c.expected_reward - c.penalty, c.progress,
                                            -c.steps, -c.action), default=None)

    def _select_recovery_waypoint(self, candidates):
        # Permit negative distance progress; novelty and executed history take
        # priority here so backtracking does not immediately reverse itself.
        return max(candidates, key=lambda c: (
            self.exploration_weight * c.unexplored - c.penalty + self.slack * c.steps,
            c.progress, -c.steps, -c.action))

    def select_action(self, *, planner, sim, observation, target_positions,
                      current_distance, goal_reached, remaining_steps=None):
        """Return a waypoint ID; STOP is reserved for confirmed goal arrival."""
        self.last_decision = None
        if remaining_steps is not None and remaining_steps < 1:
            raise RuntimeError('Episode step limit reached; reset the environment')
        if goal_reached:
            self.last_decision = dict(action=self.stop_action, mode='stop')
            return self.stop_action
        horizon = self.prediction_interval
        if remaining_steps is not None:
            horizon = min(horizon, remaining_steps)
        candidates = self._search_reachable_waypoints(planner, observation, horizon)
        partial = False
        if not candidates:
            # A turn can consume the last available step before any cell is
            # reached. A reachable target may still be partially executed until
            # the environment truncates; do not substitute an early STOP.
            candidates = self._search_reachable_waypoints(
                planner, observation, max(self.prediction_interval, 3))
            partial = bool(candidates)
        if not candidates:
            self.last_decision = dict(action=None, mode='blocked')
            raise NoReachableWaypointError(
                'Planner has no reachable non-STOP waypoint in the action map; '
                'explicit observation/recovery is required')
        scored = self._score_waypoint_candidates(
            candidates, planner, sim, target_positions, current_distance)
        selected = self._select_progress_waypoint(scored)
        mode = 'progress'
        if selected is None:
            selected = self._select_recovery_waypoint(scored)
            mode = 'recovery'
        self.last_decision = dict(action=selected.action, mode=mode,
                                  waypoint_goal=selected.goal, estimated_steps=selected.steps,
                                  expected_reward=selected.expected_reward,
                                  distance_progress=selected.progress,
                                  target_distance=selected.distance,
                                  unexplored_fraction=selected.unexplored,
                                  history_penalty=selected.penalty, partial_execution=partial,
                                  candidate_count=len(candidates))
        return int(selected.action)
