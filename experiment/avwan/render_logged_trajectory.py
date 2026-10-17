"""Render selected AV-WaN test and diagnostic trajectories as top-down maps.

The test trajectories use the saved Habitat evaluation log. Diagnostic cases
include recorded waypoint and collision telemetry. Saved SoundSpaces navigation
graphs provide map backdrops; MP3D architectural meshes are not required.

Run from the repository root:
    python -m experiment.avwan.render_logged_trajectory
"""
import json
import math
import pickle
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree
from habitat.utils.visualizations import maps

from utils.visualizations import plot_top_down_map_waypoint


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "result/2026-10-16-avwan-offline-final/full_test/actor_only_latest/test_multiple"
OUTPUT = ROOT / "result/2026-10-16-avwan-offline-final/trajectories"
DIAGNOSTICS = ROOT / "result/2026-10-15-avwan-spl-improvement/diagnostics"
METADATA = ROOT.parent / "sound-spaces/data/metadata/mp3d"

CASES = [
    {
        "category": "Test · successful navigation",
        "scene": "5ZKStnWn8Zo",
        "episode_id": "16840",
    },
    {
        "category": "Test · long-distance success",
        "scene": "fzynW3qQPVF",
        "episode_id": "87924",
    },
    {
        "category": "Test · early STOP",
        "scene": "fzynW3qQPVF",
        "episode_id": "349856",
    },
    {
        "category": "Test · source not found (distance increased)",
        "scene": "gxdoqLR6rwA",
        "episode_id": "349286",
    },
    {
        "category": "Test · max-step limit",
        "scene": "fzynW3qQPVF",
        "episode_id": "328042",
    },
    {
        "category": "Test · near-source STOP failure",
        "scene": "5ZKStnWn8Zo",
        "episode_id": "11294",
    },
    {
        "category": "Diagnostics · collision-heavy failure (14 events)",
        "scene": "5LpN3gDmAk7",
        "episode_id": "12767",
        "diagnostics": True,
    },
    {
        "category": "Diagnostics · unreachable waypoints (5)",
        "scene": "ac26ZMwG7aT",
        "episode_id": "16737",
        "diagnostics": True,
    },
]

SOURCE_RE = re.compile(
    r"setAudioSourceTransform.*?Vector\(([-+\d.eE]+),\s*([-+\d.eE]+),\s*([-+\d.eE]+)\)"
)
LISTENER_RE = re.compile(
    r"setAudioListenerTransform.*?position \[ Vector\(([-+\d.eE]+),\s*([-+\d.eE]+),\s*([-+\d.eE]+)\)"
)
RESULT_RE = re.compile(r"\d+/1000\s+(\S+)/(\S+)\s+success=(\w+)")


def read_evaluation_records(path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(row["scene"], str(row["episode_id"])): row for row in rows}


def read_logged_episodes(path):
    """Recover source and listener transforms, keyed by scene/episode."""
    episodes = []
    current = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = SOURCE_RE.search(line)
            if match:
                current = {
                    "sound_source": [float(value) for value in match.groups()],
                    "listener_positions": [],
                }
                episodes.append(current)
                continue
            if current is None:
                continue
            match = LISTENER_RE.search(line)
            if match:
                current["listener_positions"].append(
                    [float(value) for value in match.groups()]
                )
            match = RESULT_RE.search(line)
            if match:
                current["key"] = (match.group(1), str(match.group(2)))

    keyed = {episode["key"]: episode for episode in episodes if "key" in episode}
    if len(keyed) != 1000:
        raise ValueError(
            f"Expected 1,000 complete test episodes in {path}; found {len(keyed)}"
        )
    return keyed


def recorded_path(episode):
    # The first listener transform is emitted before Habitat resets the agent
    # to this episode's start pose. Exclude it, then collapse duplicate audio
    # callbacks while retaining every distinct X/Z position.
    positions = episode["listener_positions"][1:]
    path = []
    previous = None
    for position in positions:
        key = (round(position[0], 3), round(position[2], 3))
        if key != previous:
            path.append(position)
            previous = key
    if not path:
        raise ValueError("No listener positions recorded for episode")
    return path


def sample_checkpoints(path, actions):
    """Sample a readable set of checkpoints from the observed path."""
    count = min(max(1, len(actions) - 1), 12, max(1, len(path) - 1))
    if count <= 0 or len(path) < 2:
        return []
    xz = np.asarray([[point[0], point[2]] for point in path], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(xz, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] == 0:
        indices = np.linspace(1, len(path) - 1, count).round().astype(int)
    else:
        targets = np.linspace(0.0, cumulative[-1], count + 1)[1:]
        indices = np.searchsorted(cumulative, targets, side="left")
        indices = np.clip(indices, 1, len(path) - 1)
    sampled = []
    seen = set()
    for index in np.unique(indices):
        point = path[index]
        key = (round(point[0], 2), round(point[2], 2))
        if key not in seen:
            sampled.append(point)
            seen.add(key)
    return sampled


def coordinate_grid(path, source, checkpoints, nav_graph_nodes=None,
                    nav_graph_edges=None, collision_positions=None):
    """Build a Habitat-style categorical top-down map from a saved nav graph.

    The graph gives walkable connectivity but not walls, so graph edges are
    rasterized as corridors in the categorical map and Habitat's own
    ``colorize_topdown_map`` renders that map.
    """
    if not nav_graph_nodes:
        raise ValueError("No navigable graph nodes are available for this scene")
    graph_points = np.asarray(list(nav_graph_nodes.values()), dtype=np.float64)
    x_min, z_min = graph_points[:, 0].min() - 1.0, graph_points[:, 2].min() - 1.0
    x_max, z_max = graph_points[:, 0].max() + 1.0, graph_points[:, 2].max() + 1.0
    meters_per_pixel = max(x_max - x_min, z_max - z_min) / 500.0
    height = max(2, int(math.ceil((z_max - z_min) / meters_per_pixel)))
    width = max(2, int(math.ceil((x_max - x_min) / meters_per_pixel)))
    map_values = np.full((height, width), maps.MAP_INVALID_POINT, dtype=np.uint8)

    def map_coord(position):
        row = int(np.clip(round((position[2] - z_min) / meters_per_pixel), 0, height - 1))
        col = int(np.clip(round((position[0] - x_min) / meters_per_pixel), 0, width - 1))
        return [row, col]

    corridor_width = max(2, int(round(0.8 / meters_per_pixel)))
    for edge in nav_graph_edges or []:
        p1, p2 = map_coord(edge[0]), map_coord(edge[1])
        cv2.line(map_values, (p1[1], p1[0]), (p2[1], p2[0]),
                 int(maps.MAP_VALID_POINT), corridor_width, cv2.LINE_8)
    node_radius = max(1, int(round(0.4 / meters_per_pixel)))
    for point in nav_graph_nodes.values():
        row, col = map_coord(point)
        cv2.circle(map_values, (col, row), node_radius,
                   int(maps.MAP_VALID_POINT), -1, cv2.LINE_8)
    maps._outline_border(map_values)

    info = {
        "top_down_map": {
            "map": map_values,
            "fog_of_war_mask": None,
            "agent_map_coord": None,
            "agent_angle": 0.0,
            "start_map_coord": map_coord(path[0]),
            "sound_source_map_coords": [map_coord(source)],
            "waypoint_map_coords": [map_coord(point) for point in checkpoints],
            "collision_map_coords": [map_coord(point)
                                     for point in (collision_positions or [])],
            "agent_path_map_coords": [map_coord(point) for point in path],
        }
    }
    extent = (x_min, x_max, z_min, z_max)
    return info, extent


def render_case(case, record, logged_episode, output_dir):
    if case.get("diagnostics"):
        path, source, checkpoints, record, collision_positions = diagnostic_trajectory(case)
        trace_source = "Saved diagnostic rollout (train split; separate checkpoint)."
    else:
        path = recorded_path(logged_episode)
        source = logged_episode["sound_source"]
        checkpoints = sample_checkpoints(path, record["actions"])
        collision_positions = []
        trace_source = "Saved selected-policy test rollout."
    rendered_checkpoints = checkpoints
    if case.get("diagnostics") and len(checkpoints) > 12:
        indices = np.linspace(0, len(checkpoints) - 1, 12).round().astype(int)
        rendered_checkpoints = [checkpoints[index] for index in indices]
    nav_graph_nodes, nav_graph_edges = load_nav_graph_layer(case["scene"], path)
    info, extent = coordinate_grid(
        path, source, rendered_checkpoints, nav_graph_nodes=nav_graph_nodes,
        nav_graph_edges=nav_graph_edges, collision_positions=collision_positions)
    image = plot_top_down_map_waypoint(info)

    status = "Success" if record["success"] else "Failure"
    title = (
        f"{case['category']} · {case['scene']}/{case['episode_id']}\n"
        f"{status} · SPL {record['spl']:.3f} · "
        f"distance {record['initial_distance']:.0f}→{record['final_distance']:.0f} m · "
        f"{record['primitive_steps']} primitive steps"
    )
    map_pixels = image.shape[0]
    side_inches = min(10.0, max(6.0, map_pixels / 100.0))
    figure, axis = plt.subplots(
        figsize=(side_inches, side_inches * image.shape[0] / image.shape[1]), dpi=180)
    axis.imshow(image, interpolation="nearest")
    axis.axis("off")
    axis.set_title(title, fontsize=12, pad=12)
    handles = []
    if nav_graph_edges:
        handles.append(Line2D([0], [0], marker="s", color="none",
                              markerfacecolor="#969696", markersize=7,
                              label="Graph-derived navigable area"))
    if collision_positions:
        handles.append(Line2D([0], [0], marker="x", color="#d223a0", lw=0,
                              markersize=8, label="Collision event"))
    handles.extend([
        Line2D([0], [0], color="#1e96a5", lw=2, label="Recorded agent trace"),
        Line2D([0], [0], color=(1.0, 190 / 255, 0), lw=2,
               label="Start → checkpoints"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#00e600",
               markeredgecolor="white", markersize=9, label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red",
               markeredgecolor="white", markersize=8,
               label="Checkpoints (numbered)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="blue",
               markeredgecolor="white", markersize=8, label="Sound source"),
    ])
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                  fontsize=8.5, bbox_to_anchor=(0.5, 0.105))
    graph_note = ("Base map is a Habitat-colorized raster from the saved nav graph; "
                  "walls are unavailable. ")
    figure.text(0.5, 0.025,
                f"{trace_source} {graph_note}MP3D architectural mesh unavailable.",
                ha="center", va="bottom", fontsize=8, color="#555555")
    figure.tight_layout(rect=(0, 0.23, 1, 0.94))

    stem = (f"{record.get('split', 'test_multiple')}_{case['scene']}_"
            f"{case['episode_id']}")
    panel_path = output_dir / f"{stem}_annotated.png"
    figure.savefig(panel_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)
    from PIL import Image
    map_dir = output_dir / "topdownmaps"
    map_dir.mkdir(parents=True, exist_ok=True)
    image_path = map_dir / f"{case['scene']}_{case['episode_id']}.png"
    Image.fromarray(image).save(image_path)

    metadata = {
        "category": case["category"],
        "split": record.get("split", "test_multiple"),
        "scene": case["scene"],
        "episode_id": str(case["episode_id"]),
        "success": bool(record["success"]),
        "spl": float(record["spl"]),
        "initial_distance": float(record["initial_distance"]),
        "final_distance": float(record["final_distance"]),
        "primitive_steps": int(record["primitive_steps"]),
        "high_level_action_count": len(record.get("actions", [])),
        "actions": record.get("actions", []),
        "last_action": (record["actions"][-1]
                        if record.get("actions") else None),
        "collision_events": record.get("primitive_collision_events"),
        "collision_position_count": len(collision_positions),
        "unreachable_waypoints": record.get("unreachable_waypoints"),
        "navigation_graph_edges": len(nav_graph_edges),
        "rendered_waypoint_count": len(rendered_checkpoints),
        "truncated": bool(record.get("truncated", False)),
        "trace_position_count": len(path),
        "start_position_xyz": path[0],
        "sound_source_position_xyz": source,
        "trajectory_checkpoints_xyz": checkpoints,
        "checkpoint_note": (
            "Sampled from the test trace; original policy waypoint coordinates were not saved."
            if not case.get("diagnostics") else
            "Recorded policy waypoint coordinates; at most 12 are shown for readability."
        ),
        "map_note": "Habitat-colorized graph-derived occupancy proxy; MP3D mesh unavailable.",
        "trace_provenance": trace_source,
        "image": str(image_path.relative_to(ROOT)),
        "annotated_panel": str(panel_path.relative_to(ROOT)),
    }
    metadata_path = image_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    snapshot = {
        "metadata": metadata,
        "info": info,
        "rendered_map": image,
        "extent": extent,
        "path_xyz": path,
        "sound_source_xyz": source,
        "waypoints_xyz": checkpoints,
    }
    return metadata, snapshot


def load_nav_graph_layer(scene, path):
    """Load nodes and edges from the graph layer that matches the trajectory."""
    graph_path = METADATA / scene / "graph.pkl"
    if not graph_path.exists():
        return {}, []
    with graph_path.open("rb") as handle:
        graph = pickle.load(handle)
    graph_nodes = {
        node: attributes["point"]
        for node, attributes in graph.nodes(data=True)
        if attributes.get("point") is not None
    }
    points = np.asarray(list(graph_nodes.values()), dtype=np.float64)
    path_xz = np.asarray(path, dtype=np.float64)[:, [0, 2]]
    layer_values = np.round(points[:, 1], 1)
    layer_scores = {}
    for layer in np.unique(layer_values):
        layer_points = points[layer_values == layer][:, [0, 2]]
        layer_scores[layer] = float(cKDTree(layer_points).query(path_xz)[0].mean())
    selected_layer = min(layer_scores, key=layer_scores.get)
    nodes = {
        node: point for node, point in graph_nodes.items()
        if abs(round(float(point[1]), 1) - selected_layer) <= 0.4
    }
    edges = [(nodes[a], nodes[b]) for a, b in graph.edges()
             if a in nodes and b in nodes]
    return nodes, edges


def diagnostic_trajectory(case):
    """Read collision and waypoint telemetry from saved diagnostic metadata."""
    trace_path = DIAGNOSTICS / "metadata" / case["scene"] / f"{case['episode_id']}.json"
    dataset_path = DIAGNOSTICS / "dataset.json"
    metadata = json.loads(trace_path.read_text())
    episodes = json.loads(dataset_path.read_text())["episodes"]
    episode = next(ep for ep in episodes
                   if Path(ep["scene_id"]).stem == case["scene"]
                   and str(ep["episode_id"]) == str(case["episode_id"]))
    path = [episode["start_position"]]
    path.extend(step["agent_position_after"] for step in metadata["trace"])
    source = episode["goals"][0]["position"]
    checkpoints = [step["waypoint_position"] for step in metadata["trace"]]
    collision_positions = [step["agent_position_after"] for step in metadata["trace"]
                           if int(step.get("collision_count", 0)) > 0]
    record = {
        "split": metadata["split"],
        "success": bool(metadata["success"]),
        "spl": float(metadata["spl"]),
        "initial_distance": float(metadata["initial_distance"]),
        "final_distance": float(metadata["final_distance"]),
        "primitive_steps": int(metadata["primitive_steps"]),
        "actions": metadata["actions"],
        "truncated": bool(metadata["truncated"]),
        "primitive_collision_events": int(metadata["primitive_collision_events"]),
        "unreachable_waypoints": int(metadata["unreachable_waypoints"]),
    }
    return path, source, checkpoints, record, collision_positions


def main():
    records_path = EVALUATION / "episodes.jsonl"
    log_path = EVALUATION.with_suffix(".log")
    records = read_evaluation_records(records_path)
    logged = read_logged_episodes(log_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rendered, snapshots = [], []
    for case in CASES:
        if case.get("diagnostics"):
            result, snapshot = render_case(case, None, None, OUTPUT)
        else:
            key = (case["scene"], str(case["episode_id"]))
            if key not in records or key not in logged:
                raise KeyError(f"Episode {key[0]}/{key[1]} is missing from the test records/log")
            result, snapshot = render_case(case, records[key], logged[key], OUTPUT)
        rendered.append(result)
        snapshots.append(snapshot)
    (OUTPUT / "logged_test_cases.json").write_text(
        json.dumps(rendered, indent=2) + "\n"
    )
    with (OUTPUT / "topdown_snapshots.pkl").open("wb") as handle:
        pickle.dump(snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(snapshots)} map snapshots to {OUTPUT / 'topdown_snapshots.pkl'}")


if __name__ == "__main__":
    main()
