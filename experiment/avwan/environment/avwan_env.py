"""AV-WaN waypoint environment for offline data collection and policy evaluation.

The planner is provided by SoundSpaces. Unlike its PPO MapNavEnv, this wrapper
never resets inside step(): the returned observation belongs to the same episode.
"""
import sys
sys.path.append("/home/getuanhui/project/habitat")
from copy import deepcopy
import math
from numbers import Integral
from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from gym import spaces
import numpy as np
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner
from soundspaces.datasets.audionav_dataset import AudioNavDataset
from collections import deque
import networkx as nx
from habitat.utils.geometry_utils import quaternion_rotate_vector

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AVWANRLEnv")
class AVWANRLEnv(habitat.RLEnv):
    """One action selects a row-major cell of the egocentric action map.

    The center cell means STOP (40 for a 9x9 map). step() retains the Habitat
    four-value API and returns the undiscounted sum of primitive rewards.
    step_transition() / last_transition expose SMDP data for offline training:
    reward=sum(gamma**i * r_i), discount=gamma**num_steps unless terminated.
    Time limits truncate an episode but keep a nonzero bootstrap discount.
    """

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._continuous = bool(getattr(config, "CONTINUOUS", False))
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE
        self._map_size = config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE
        self._prediction_interval = config.PREDICTION_INTERVAL
        self.gamma = float(getattr(getattr(config, "OFFLINE_RL", None), "GAMMA", 0.99))
        if not isinstance(self._map_size, Integral) or self._map_size < 3 or self._map_size % 2 != 1:
            raise ValueError("ACTION_MAP.MAP_SIZE must be an odd integer >= 3")
        if not isinstance(self._prediction_interval, Integral) or self._prediction_interval < 1:
            raise ValueError("PREDICTION_INTERVAL must be a positive integer")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("OFFLINE_RL.GAMMA must be in [0, 1]")
        required = {"EGOMAP_SENSOR", "COLLISION", "ACTION_MAP"}
        if not required.issubset(config.TASK_CONFIG.TASK.SENSORS):
            raise ValueError(f"AV-WaN needs sensors {sorted(required)}")
        self.stop_action = self._map_size ** 2 // 2
        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._needs_reset = True
        self._last_transition = None
        super().__init__(self._core_env_config, dataset)
        # Habitat's primitive action space has four entries; the policy selects waypoints.
        self.action_space = spaces.Discrete(self._map_size ** 2)
        if isinstance(self.observation_space, spaces.Dict):
            self.observation_space = deepcopy(self.observation_space)
            for key in ("gm", "am", "action_map"):
                if key in self.observation_space.spaces:
                    shape = self.observation_space.spaces[key].shape
                    self.observation_space.spaces[key] = spaces.Box(
                        low=0, high=np.inf if key == "am" else 1, shape=shape, dtype=np.float32)
        self.planner = Planner(model_dir=getattr(config, "MODEL_DIR", None),
                               use_acoustic_map="ACOUSTIC_MAP" in config.TASK_CONFIG.TASK.SENSORS,
                               masking=getattr(config, "MASKING", True),
                               task_config=config.TASK_CONFIG)

    def reset(self):
        self._needs_reset = True
        self._last_transition = None
        self._previous_action = None
        self.planner.reset()
        observations = self._normalize_observation(super().reset())
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_target_distance = self._distance_target()
        self._previous_observation = deepcopy(observations)
        self._needs_reset = False
        return observations

    @staticmethod
    def _normalize_observation(observations):
        """Match policy sensor shapes for this SoundSpaces fork and cached depth."""
        observations = dict(observations)
        if "depth" in observations:
            depth = np.asarray(observations["depth"], dtype=np.float32)
            while depth.ndim > 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim == 2:
                depth = depth[..., None]
            if depth.ndim != 3 or depth.shape[-1] != 1:
                raise ValueError(f"Expected HWC depth with one channel, got {depth.shape}")
            observations["depth"] = depth
        if "spectrogram" in observations:
            audio = observations["spectrogram"]
            if isinstance(audio, tuple) and len(audio) == 2:
                audio = audio[0]
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 4 and audio.shape[0] == 1:
                audio = audio[0]
            if audio.ndim != 3 or audio.shape[-1] != 2:
                raise ValueError(f"Expected FxTx2 spectrogram, got {audio.shape}")
            observations["spectrogram"] = audio
        return observations

    def _parse_action(self, action):
        if isinstance(action, dict):
            if set(action) != {"action"}:
                raise ValueError("Expected a waypoint ID or {'action': waypoint_id}")
            action = action["action"]
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
            raise ValueError("Waypoint action must be an integer cell index")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Waypoint action must be in [0, {self.action_space.n - 1}]")
        return action

    def action_mask(self, observation=None):
        """AV-WaN's local occupancy mask; it is not a guarantee of graph reachability."""
        if observation is None:
            observation = self._previous_observation
        if observation is None:
            raise RuntimeError("Call reset() before requesting an action mask")
        mask = np.asarray(observation["action_map"]).reshape(-1) > 0.5
        if mask.size != self.action_space.n:
            raise ValueError("action_map shape does not match the waypoint action space")
        mask[self.stop_action] = True
        return mask

    def get_expert_map_action(self):
        """Return the best local waypoint ID under shortest-path reward scoring.

        BFS searches (cell, heading), counting both turns and forward moves in
        the discrete SoundSpaces navigation graph. Scores use the undiscounted
        reward returned by step(), not last_transition's discounted reward.
        This is an oracle over known graph paths; the planner's later choices
        and map updates can produce a different executed path.
        """


        if self._needs_reset:
            raise RuntimeError("Call reset() before requesting an expert action")
        sim = self.habitat_env.sim
        if self._continuous:
            raise NotImplementedError("Expert map BFS requires discrete SoundSpaces navigation")

        size = self._map_size
        center = size // 2
        mask = self.action_mask()
        agent_state = sim.get_agent_state()
        origin = np.asarray(agent_state.position, dtype=np.float64)
        resolution = float(self._core_env_config.TASK.ACTION_MAP.MAP_RESOLUTION)
        horizon = self._prediction_interval
        max_steps = self.habitat_env._max_episode_steps
        if max_steps:
            horizon = min(horizon, max_steps - self.habitat_env._elapsed_steps)
        if horizon < 1:
            raise RuntimeError("Episode step limit reached; call reset()")

        # Convert the egocentric grid to simulator positions. Mapper coordinates
        # are relative to the initial pose, and are NOT simulator world positions.
        cells = {}
        for action in range(size * size):
            if not mask[action]:
                continue
            row, col = divmod(action, size)
            offset = np.array([(col - center) * resolution, 0.0,
                               (row - center) * resolution])
            position = origin + quaternion_rotate_vector(agent_state.rotation, offset)
            try:
                node = sim._position_to_index(position.tolist())
            except ValueError:
                # An apparently free/unexplored map cell may not exist in the
                # simulator's ground-truth navigation graph.
                continue
            goal = self.planner.get_map_coordinates(action)
            planner_node = self.planner._map_index_to_graph_nodes([goal])[0]
            if node in sim.graph and planner_node in self.planner._graph:
                cells[action] = (node, planner_node)

        current_distance = self._distance_target()
        previous_distance = self._previous_target_distance
        slack = float(self._rl_config.SLACK_REWARD) if self._rl_config.WITH_TIME_PENALTY else 0.0
        scale = float(self._rl_config.DISTANCE_REWARD_SCALE) if self._rl_config.WITH_DISTANCE_REWARD else 0.0

        # STOP consumes one primitive step, including when it fails. Merely
        # reaching the sound source with a movement action earns no success bonus.
        stop_reward = slack + scale * (previous_distance - current_distance)
        if sim.reaching_goal:
            stop_reward += self._rl_config.SUCCESS_REWARD
        best_action = self.stop_action
        best_key = (stop_reward, -current_distance, -1, -best_action)
        if self.stop_action not in cells:
            return best_action

        # In the current egocentric frame: right, down, left, up. Initially the
        # agent faces up. Each BFS edge costs exactly one primitive action.
        directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
        queue = deque([(self.stop_action, 3, 0)])
        visited = {(self.stop_action, 3)}
        scored = {self.stop_action}
        targets = [goal.position for goal in self.current_episode.goals]
        while queue:
            action, heading, steps = queue.popleft()
            if action not in scored:
                scored.add(action)
                node = cells[action][0]
                position = sim.graph.nodes[node]["point"]
                try:
                    distance = float(sim.geodesic_distance(position, targets))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    distance = float("inf")
                if math.isfinite(distance):
                    # Distance progress telescopes across all primitive steps;
                    # rotations contribute only their time penalty.
                    reward = steps * slack + scale * (previous_distance - distance)
                    key = (reward, -distance, -steps, -action)
                    if key > best_key:
                        best_action, best_key = action, key
            if steps >= horizon:
                continue

            successors = [(action, (heading - 1) % 4),
                          (action, (heading + 1) % 4)]
            row, col = divmod(action, size)
            dr, dc = directions[heading]
            next_row, next_col = row + dr, col + dc
            if 0 <= next_row < size and 0 <= next_col < size:
                next_action = next_row * size + next_col
                if next_action in cells:
                    node, planner_node = cells[action]
                    next_node, next_planner_node = cells[next_action]
                    # Free endpoints alone do not imply a traversable edge.
                    if (sim.graph.has_edge(node, next_node)
                            and self.planner._graph.has_edge(planner_node, next_planner_node)):
                        successors.append((next_action, heading))
            for next_state in successors:
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((*next_state, steps + 1))

        return int(best_action)
    
    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), including after episode termination")
        action = self._parse_action(action)
        observation = deepcopy(self._previous_observation)
        state = deepcopy(observation)
        episode_id = str(self.current_episode.episode_id)
        scene_id = str(self.current_episode.scene_id)
        self._previous_action = action
        goal = self.planner.get_map_coordinates(action)
        # A simulator/planner exception may leave a partially executed macro-action.
        # Require a reset after such an error rather than recording a broken transition.
        self._needs_reset = True
        self._last_transition = None
        stop = action == self.stop_action
        rewards, primitive_actions = [], []
        reaching_waypoint = cant_reach_waypoint = done = False
        info = {}
        for step_index in range(self._prediction_interval):
            # Preserve the official first-step fallback for blocked/unreachable goals.
            # This ensures every stored transition has at least one simulator step.
            if step_index and not stop and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            primitive = int(self.planner.plan(observation, goal, stop=stop))
            observation, reward, done, info = super().step({"action": primitive})
            observation = self._normalize_observation(observation)
            rewards.append(float(reward))
            primitive_actions.append(primitive)
            # Update even on termination: next_state needs the FINAL map and pose.
            self.planner.update_map_and_graph(observation)
            x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
            reaching_waypoint = not stop and (x, y) == tuple(goal)
            if done or reaching_waypoint:
                break
        self.planner.add_maps_to_observation(observation)
        if not stop and not reaching_waypoint:
            cant_reach_waypoint |= not self.planner.check_navigability(goal)

        success = self._episode_success()
        # Habitat sets episode_over for both task termination and time limits.
        task_ended = not self.habitat_env.task.is_episode_active
        terminated = bool(done and (success or self.habitat_env.task.is_stop_called or task_ended))
        truncated = bool(done and not terminated)
        duration = len(rewards)
        discounted_reward = sum(self.gamma ** i * r for i, r in enumerate(rewards))
        discount = 0.0 if terminated else self.gamma ** duration
        info = deepcopy(info)
        info.update(episode_id=episode_id, scene_id=scene_id, waypoint_action=action,
                    waypoint_goal=tuple(goal), reaching_waypoint=bool(reaching_waypoint),
                    cant_reach_waypoint=bool(cant_reach_waypoint), success=bool(success),
                    terminated=terminated, truncated=truncated, num_steps=duration,
                    discounted_reward=discounted_reward, discount=discount,
                    low_level_actions=primitive_actions, low_level_rewards=rewards)
        info["TimeLimit.truncated"] = truncated
        self._last_transition = dict(
            observation=state, action=action, reward=discounted_reward,
            next_observation=deepcopy(observation), terminated=terminated, truncated=truncated,
            done=bool(done), num_steps=duration, discount=discount, gamma=self.gamma,
            undiscounted_reward=sum(rewards), low_level_actions=primitive_actions.copy(),
            low_level_rewards=rewards.copy(), episode_id=episode_id, scene_id=scene_id,
            action_mask=self.action_mask(state), next_action_mask=self.action_mask(observation))
        self._previous_observation = deepcopy(observation)
        self._needs_reset = bool(done)
        return observation, sum(rewards), bool(done), info

    @property
    def last_transition(self):
        if self._last_transition is None:
            raise RuntimeError("No transition available; call step() first")
        return deepcopy(self._last_transition)

    def step_transition(self, action):
        """Collect one owned transition, ready for save_transitions()."""
        self.step(action)
        return self.last_transition

    def get_reward_range(self):
        # Geodesic progress and the multi-step return have no fixed +/-1 bound.
        return (-float("inf"), float("inf"))

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD if self._rl_config.WITH_TIME_PENALTY else 0.0
        current = self._distance_target()
        if self._rl_config.WITH_DISTANCE_REWARD:
            reward += (self._previous_target_distance - current) * self._rl_config.DISTANCE_REWARD_SCALE
        self._previous_target_distance = current
        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        if not math.isfinite(reward):
            raise ValueError(f"Non-finite AV-WaN reward: {reward}")
        return float(reward)

    def _distance_target(self):
        distance = self.habitat_env.get_metrics().get("distance_to_goal")
        if distance is None:
            sim = self.habitat_env.sim
            distance = sim.geodesic_distance(sim.get_agent_state().position.tolist(),
                                             [g.position for g in self.current_episode.goals])
        distance = float(distance)
        if not math.isfinite(distance):
            raise ValueError(f"Non-finite distance_to_goal: {distance}")
        return distance

    def _episode_success(self):
        if not self.habitat_env.task.is_stop_called:
            return False
        if self._continuous:
            return self._distance_target() < self._success_distance
        return bool(self.habitat_env.sim.reaching_goal)

    def get_done(self, observations):
        return bool(self.habitat_env.episode_over or self._episode_success())

    def get_info(self, observations):
        return deepcopy(self.habitat_env.get_metrics())

    def get_current_episode_id(self):
        return self.current_episode.episode_id

    def global_to_egocentric(self, point):
        return self.planner.mapper.global_to_egocentric(*point)

    def egocentric_to_global(self, point):
        return self.planner.mapper.egocentric_to_global(*point)
