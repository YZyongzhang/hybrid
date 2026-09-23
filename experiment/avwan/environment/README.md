# Local AV-WaN navigation

`AVWANRLEnv` uses these project-owned implementations:

- `Planner.py`: waypoint-to-primitive planning, graph updates and action masks.
- `Mapper.py`: geometric/acoustic maps and coordinate transforms.
- `nav.py`: the `LocalEgoMap` depth sensor, with observation UUID `ego_map`,
  plus local geometric/action/acoustic map placeholders and intensity sensor.
- `BFSExpert.py`: expert waypoint selection on the planner's current graph.

Planner and Mapper were copied and adapted from SoundSpaces
`ss_baselines/av_wan/models/{planner,mapper}.py`. The local `nav.py` contains
the relevant EgoMap implementation from `soundspaces/tasks/nav.py` and map
sensor definitions from `ss_baselines/av_wan/avwan_sensors.py`; unrelated
task/sensor classes remain supplied by SoundSpaces. These copies retain their
upstream copyright notice and license (`SOUNDSPACES_LICENSE`). Upstream:
https://github.com/facebookresearch/sound-spaces.

## Behavior changes

The mapper immediately records occupied cells. Two subsequent observations of
an explored, unoccupied cell clear that obstacle; an occupied observation
resets its free-observation counter. Unknown cells do not clear obstacles.
Exploration history remains cumulative. The threshold of two is local to
`Mapper.update`, and reset clears all map evidence.

The depth sensor keeps the original obstacle height thresholds. It rounds depth
points to the nearest cell center, matching the mapper's metric coordinates;
floor-only quantization could shift nearby furniture onto a navigation edge.
Neither channel uses morphological closing. Closing the obstacle channel can
fabricate occupied evidence at a measured floor cell or an unobserved gap, then
overwrite known free space after a turn. Missing returns now preserve the map's
existing evidence; actual obstacle returns still block cells, including cells
that also contain floor returns. The sensor projects each depth observation
locally, without reading/writing the official EgoMap cache.

The planner restores depth-blocked nodes/edges when the corresponding map cells
become free. Restoration uses the original cardinal grid topology, not the
simulator's ground-truth navigation graph. A failed forward action independently
blocks the attempted edge in both directions for the rest of the episode.
The destination node remains available through other routes. Depth is still
integrated after a collision without advancing the map pose, but free depth
observations and node restoration cannot reopen a collision-blocked edge.
Reset clears collision blocks. This rule assumes the static, discrete
SoundSpaces graph: a failed forward action is evidence of a missing transition.
In the discrete simulator, depth-only blocks can be tested via one adjacent
forward transition. Collision-blocked edges cannot be probed. Successful moves
record verified edges and endpoint nodes; later depth occupancy cannot erase
this executed connectivity, while collision evidence still takes precedence.
Raw depth occupancy remains unchanged: an observed obstacle and a verified
simulator transition can disagree. Reset clears all verified transitions.
This recovery is disabled by default for continuous simulators, whose forward
actions need not reach the next grid node.

With `MASKING=True`, `action_map` includes connected waypoints, STOP, and these
adjacent depth probes. A probe is an allowed attempt, not a guarantee of free
space. `check_navigability` still checks graph reachability;
`check_waypoint_executability` also accepts a one-edge probe. Turning toward
the probe does not cancel it. If a collision leaves no route to the waypoint,
the environment ends the macro-action and reports `cant_reach_waypoint`.
The expert first selects normal progress waypoints, then progress probes, then
exploration/recovery. Its decision metadata marks probes with `depth_probe`.

The environment clones the caller's config and selects `LocalEgoMap` before
constructing Habitat. The official `EgoMap` registry entry is left unchanged.
The waypoint action IDs and observation shapes are unchanged, but map/mask
values differ; data collection, training and evaluation should use this same
local environment for consistent behavior.

These changes address irreversible obstacle accumulation and repeated attempts
to traverse failed edges. They do not guarantee globally shortest expert
trajectories, or make the projected depth map identical to the simulator graph.
No goal-specific clearing, obstacle expiration or ground-truth graph repair is
used.

## Validation

Run from the repository root with the `ss` Python environment:

```bash
python -m unittest discover -s tests -p test_local_avwan_navigation.py -v
MPLBACKEND=Agg python tests/run_local_avwan_validation.py
```

The second command requires Habitat GPU access and runs three fixed episodes
in separate processes. Results and top-down maps are written to
`outputs/local_navigation_depth_fix/`; earlier results are preserved. The runner
checks that the actual Planner, Mapper and EgoMap instances are local, verifies
expert/reset lifecycle behavior, and requires successful arrival for episodes
`r1Q1Z4BcV1o/77250` and `GdvgFV5R1Z5/11`. The latter now reaches the source and
STOPs in 10 primitive actions with no collisions (previously 200-step timeout).
For episode `VVfe2KiqLaN/548`, a depth observation blocked valid approaches to
the source. Later close views had no depth returns in those cells, so the
two-free-observation rule never cleared them. Collision blocks alone did not
solve this. One-edge probing and verified transitions allow actual arrival;
with seed 20260924 and prediction interval 10, the focused regression takes
37 primitive actions with one collision and ends at distance zero.

For a focused collision regression with per-primitive records and Habitat's
standard TopDownMap rendering, run:

```bash
python tests/run_collision_recovery.py
python tests/run_collision_recovery.py --scene r1Q1Z4BcV1o --episode 77250 --seed 0 --require-success
python tests/run_collision_recovery.py --scene GdvgFV5R1Z5 --episode 11 --seed 20260926 --require-success
```

Outputs go to `tests/results/collision_recovery/`. The runner checks that a
collision immediately removes the attempted edge, no undirected edge collides
twice, and reset clears collision history. It also reports arrival and the
source's map state when the source lies in the local waypoint window.
`--require-success` additionally asserts arrival; without it, a passing collision
regression does not imply navigation success.

For a reproducible sample across 12 scenes (three regressions plus nine sampled
short/medium/long routes), run:

```bash
MPLBACKEND=Agg python tests/run_waypoint_episode_batch.py --output outputs/waypoint_batch --count 12
```

The batch saves its selection, source hashes, per-episode traces/maps/logs, and
an incremental `summary.json`. It checks observation/reward validity, action
masks, waypoint positions, transition metadata, and reset lifecycle. Navigation
success, step-limit timeout, no reachable waypoint, and runtime errors have
separate statuses. The batch continues after a failure and exits nonzero unless
every sampled episode succeeds; a sample does not establish full-dataset success.
