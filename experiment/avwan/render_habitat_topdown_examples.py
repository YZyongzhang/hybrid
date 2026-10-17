"""Render selected saved AV-WaN traces over Habitat's MP3D top-down maps.

Run from the repository root with the SoundSpaces environment active:

    python -m experiment.avwan.render_habitat_topdown_examples
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-avwan-topdown")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from habitat.utils.visualizations import maps

from experiment import AVWANRLEnv
from experiment.avwan.config.default import get_config
from experiment.avwan.evaluate import load_episodes
from experiment.avwan.render_logged_trajectory import (
    DIAGNOSTICS,
    EVALUATION,
    diagnostic_trajectory,
    read_evaluation_records,
    read_logged_episodes,
    recorded_path,
    sample_checkpoints,
)
from utils.visualizations import plot_top_down_map_waypoint


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "result/10-11/total/2026-10-04-stop-negative-recovery-study/evaluation.yaml"
OUTPUT = ROOT / "result/2026-10-16-avwan-offline-final/trajectories/habitat_topdownmaps"
EXAMPLES = [
    ("5ZKStnWn8Zo", "16840"),   # successful navigation
    ("fzynW3qQPVF", "87924"),   # long-distance success
    ("fzynW3qQPVF", "349856"),  # early STOP
    ("5LpN3gDmAk7", "12767"),   # collision-heavy diagnostic
]
LABELS = {
    ("5ZKStnWn8Zo", "16840"): "Successful navigation",
    ("fzynW3qQPVF", "87924"): "Long-distance success",
    ("fzynW3qQPVF", "349856"): "Early STOP failure",
    ("5LpN3gDmAk7", "12767"): "Collision-heavy failure (14 events)",
}


def _episode_from_record(record, dataset_config):
    from soundspaces.datasets.audionav_dataset import AudioNavDataset

    dataset = AudioNavDataset()
    dataset._config = dataset_config
    dataset.from_json(
        json.dumps({"episodes": [record]}),
        scenes_dir=dataset_config.SCENES_DIR,
    )
    if len(dataset.episodes) != 1:
        raise ValueError("Could not parse the requested episode")
    return dataset.episodes[0]


def _world_to_map(info, sim, measure, positions):
    full_shape = measure._top_down_map.shape[:2]
    map_info = info["top_down_map"]
    agent_position = sim.get_agent_state().position
    agent_grid = np.asarray(
        maps.to_grid(agent_position[2], agent_position[0], full_shape, sim=sim)
    )
    crop_offset = agent_grid - np.asarray(map_info["agent_map_coord"])
    coords = []
    for position in positions:
        grid = maps.to_grid(position[2], position[0], full_shape, sim=sim)
        coords.append((np.asarray(grid) - crop_offset).astype(int).tolist())
    return coords


def _get_case_trace(case, records, logged, diagnostic_records):
    scene, episode_id = case
    if (scene, episode_id) == ("5LpN3gDmAk7", "12767"):
        descriptor = {"scene": scene, "episode_id": episode_id, "diagnostics": True}
        path, source, checkpoints, result, collisions = diagnostic_trajectory(descriptor)
        raw = next(
            episode for episode in diagnostic_records
            if Path(episode["scene_id"]).stem == scene
            and str(episode["episode_id"]) == episode_id
        )
        return path, source, checkpoints, result, collisions, raw, "Collision-heavy failure"

    key = (scene, episode_id)
    if key not in records or key not in logged:
        raise KeyError(f"Missing saved test trace for {scene}/{episode_id}")
    result = records[key]
    episode = logged[key]
    path = recorded_path(episode)
    source = episode["sound_source"]
    checkpoints = sample_checkpoints(path, result["actions"])
    return path, source, checkpoints, result, [], None, LABELS[key]


def _font(size):
    candidate = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(candidate), size) if candidate.exists() else ImageFont.load_default()


def render():
    config = get_config(
        str(CONFIG), opts=["TASK_CONFIG.DATASET.SPLIT", "test_multiple"]
    )
    config.defrost()
    config.TASK_CONFIG.TRAIN.OPEN = False
    config.TASK_CONFIG.COLLECT.OPEN = False
    measurements = list(config.TASK_CONFIG.TASK.MEASUREMENTS)
    if "TOP_DOWN_MAP" not in measurements:
        measurements.append("TOP_DOWN_MAP")
    config.TASK_CONFIG.TASK.MEASUREMENTS = measurements
    topdown = config.TASK_CONFIG.TASK.TOP_DOWN_MAP
    topdown.MAP_RESOLUTION = 500
    topdown.FOG_OF_WAR.DRAW = False
    topdown.DRAW_SOURCE_AND_TARGET = False
    topdown.DRAW_SHORTEST_PATH = False
    config.freeze()

    dataset = load_episodes(config)
    test_episodes = {
        (Path(episode.scene_id).stem, str(episode.episode_id)): episode
        for episode in dataset.episodes
    }
    test_records = read_evaluation_records(EVALUATION / "episodes.jsonl")
    logged = read_logged_episodes(EVALUATION.with_suffix(".log"))
    diagnostic_records = json.loads((DIAGNOSTICS / "dataset.json").read_text())["episodes"]

    env = AVWANRLEnv(config, dataset)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rendered = []
    images = []
    try:
        for index, case in enumerate(EXAMPLES):
            path, source, checkpoints, result, collisions, raw_episode, label = _get_case_trace(
                case, test_records, logged, diagnostic_records
            )
            key = case
            episode = (
                _episode_from_record(raw_episode, config.TASK_CONFIG.DATASET)
                if raw_episode is not None else test_episodes[key]
            )
            env.habitat_env.current_episode = episode
            env.seed(20261030 + index)
            observation = env.reset()

            # Habitat populates TopDownMap's first metric during a step. Update
            # the measure directly so rendering the saved trace leaves the
            # episode at its recorded start pose.
            measure = env.habitat_env.task.measurements.measures["top_down_map"]
            measure.update_metric(episode, action=None)
            info = env.get_info(observation)
            info["top_down_map"]["agent_path_map_coords"] = _world_to_map(
                info, env.habitat_env.sim, measure, path
            )
            info["top_down_map"]["waypoint_map_coords"] = _world_to_map(
                info, env.habitat_env.sim, measure, checkpoints
            )
            info["top_down_map"]["collision_map_coords"] = _world_to_map(
                info, env.habitat_env.sim, measure, collisions
            )
            info["top_down_map"]["agent_map_coord"] = None
            image = plot_top_down_map_waypoint(info)

            scene, episode_id = case
            stem = f"{scene}_{episode_id}"
            image_path = OUTPUT / f"{stem}.png"
            Image.fromarray(image).save(image_path)
            images.append((label, scene, episode_id, result, Image.fromarray(image)))
            metadata = {
                "scene": scene,
                "episode_id": episode_id,
                "split": result.get("split", "train_multiple" if raw_episode else "test_multiple"),
                "success": bool(result.get("success", False)),
                "spl": float(result.get("spl", 0.0)),
                "initial_distance": float(result.get("initial_distance", 0.0)),
                "final_distance": float(result.get("final_distance", 0.0)),
                "waypoint_count": len(checkpoints),
                "collision_count": len(collisions),
                "map_source": "Habitat TOP_DOWN_MAP measure using the MP3D scene assets",
                "trajectory_source": "Saved rollout coordinates",
                "image": str(image_path.relative_to(ROOT)),
            }
            image_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            rendered.append(metadata)
            print(f"Rendered Habitat map {index + 1}/{len(EXAMPLES)}: {image_path}", flush=True)
    finally:
        env.close()

    cell_w, cell_h, header_h = 620, 410, 62
    sheet = Image.new("RGB", (cell_w * 2, (cell_h + header_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, detail_font = _font(18), _font(14)
    for slot, (label, scene, episode_id, result, image) in enumerate(images):
        row, col = divmod(slot, 2)
        x, y = col * cell_w, row * (cell_h + header_h)
        status = "Success" if result.get("success", False) else "Failure"
        detail = f"{scene}/{episode_id}  |  {status}  |  SPL {float(result.get('spl', 0.0)):.3f}"
        draw.text((x + 12, y + 7), label, font=title_font, fill="#111111")
        draw.text((x + 12, y + 34), detail, font=detail_font, fill="#555555")
        image.thumbnail((cell_w - 24, cell_h - 16), Image.Resampling.LANCZOS)
        px = x + (cell_w - image.width) // 2
        py = y + header_h + (cell_h - image.height) // 2
        sheet.paste(image, (px, py))
        draw.rectangle((x, y, x + cell_w - 1, y + header_h + cell_h - 1), outline="#dddddd")

    sheet_path = OUTPUT.parent / "topdown_examples_habitat.png"
    sheet.save(sheet_path)
    summary_path = OUTPUT.parent / "habitat_topdown_examples.json"
    summary_path.write_text(json.dumps(rendered, indent=2) + "\n")
    print(f"Saved contact sheet: {sheet_path}", flush=True)


if __name__ == "__main__":
    render()
