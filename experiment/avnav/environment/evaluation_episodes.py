"""Load SoundSpaces test episodes for AVNav's closed-loop evaluator."""
import gzip
import json
from pathlib import Path
import random


def load_episodes(dataset_config, max_episodes=0, seed=0, episode_keys=None):
    """Build a deterministic AudioNavDataset from the configured split."""
    from soundspaces.datasets.audionav_dataset import AudioNavDataset

    path = Path(dataset_config.DATA_PATH.format(
        version=dataset_config.VERSION, split=dataset_config.SPLIT
    ))
    if not path.exists() and path.suffix == ".gz" and path.with_suffix("").exists():
        path = path.with_suffix("")
    opener = gzip.open if path.suffix == ".gz" else open
    wanted = None
    if episode_keys is not None:
        wanted = {(str(scene), str(episode_id)) for scene, episode_id in episode_keys}
        if not wanted:
            raise ValueError("episode_keys must not be empty")

    with opener(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle).get("episodes", [])

    def key(episode):
        return Path(episode["scene_id"]).stem, str(episode["episode_id"])

    if wanted is not None:
        records = [episode for episode in records if key(episode) in wanted]
        found = {key(episode) for episode in records}
        missing_scenes = {scene for scene, _ in wanted - found}
        for scene in sorted(missing_scenes):
            content = path.parent / "content" / f"{scene}.json.gz"
            if not content.exists():
                plain_content = content.with_suffix("")
                if plain_content.exists():
                    content = plain_content
                else:
                    continue
            content_opener = gzip.open if content.suffix == ".gz" else open
            with content_opener(content, "rt", encoding="utf-8") as handle:
                candidates = json.load(handle).get("episodes", [])
            records.extend(episode for episode in candidates
                           if key(episode) in wanted and key(episode) not in found)
            found.update(key(episode) for episode in records if key(episode) in wanted)
        if found != wanted:
            raise ValueError(f"Missing {len(wanted - found)} requested episodes from {dataset_config.SPLIT}")
    elif not records:
        for content in sorted((path.parent / "content").glob("*.json*")):
            if content.suffix not in (".json", ".gz"):
                continue
            content_opener = gzip.open if content.suffix == ".gz" else open
            with content_opener(content, "rt", encoding="utf-8") as handle:
                records.extend(json.load(handle).get("episodes", []))

    records.sort(key=key)
    if max_episodes:
        random.Random(seed).shuffle(records)
        records = records[:max_episodes]
        records.sort(key=key)
    if not records or len({key(episode) for episode in records}) != len(records):
        raise ValueError(f"Empty or duplicate episodes for split {dataset_config.SPLIT}")

    dataset = AudioNavDataset()
    dataset._config = dataset_config
    # SoundSpaces' reader caps a single JSON load at 100 records.
    for offset in range(0, len(records), 100):
        dataset.from_json(
            json.dumps({"episodes": records[offset:offset + 100]}),
            scenes_dir=dataset_config.SCENES_DIR,
        )
    dataset.episodes.sort(key=lambda episode: (
        Path(episode.scene_id).stem, str(episode.episode_id)
    ))
    if len(dataset.episodes) != len(records):
        raise ValueError(
            f"SoundSpaces loaded {len(dataset.episodes)} of {len(records)} episodes "
            f"from {dataset_config.SPLIT}"
        )
    return dataset
