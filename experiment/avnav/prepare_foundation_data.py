"""Convert diverse AVNav collection episodes into foundation training shards."""
import argparse
import json
from pathlib import Path
import shutil

from Dataset.dataset_avnav import convert_pickle_to_pt


def _successful_source(source):
    source = Path(source).expanduser().resolve()
    index_path = source.parent / "episodes.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Collection index is required for success filtering: {index_path}")
    successful_root = source.parent / "foundation_success_episodes"
    successful_root.mkdir(parents=True, exist_ok=True)
    count = 0
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("success", False):
                continue
            original = (source.parent / row["file"]).resolve()
            if not original.is_file():
                raise FileNotFoundError(f"Successful episode is missing: {original}")
            if source not in original.parents:
                raise ValueError(f"Episode index points outside the AVNAV collection: {original}")
            link = successful_root / row["scene"] / original.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != original:
                    raise FileExistsError(f"Unexpected file in success-only source: {link}")
            elif link.is_file():
                source_stat = original.stat()
                target_stat = link.stat()
                if target_stat.st_size != source_stat.st_size:
                    shutil.copyfile(original, link)
            else:
                shutil.copyfile(original, link)
            count += 1
    if not count:
        raise ValueError("No successful episodes are available for foundation fine-tuning")
    return successful_root, count


def run(source, destination, shard_size=256, seed=20261018, success_only=True):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    conversion_source = source
    successful_episode_count = None
    if success_only:
        conversion_source, successful_episode_count = _successful_source(source)
    manifest = destination / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if Path(previous.get("source", "")).resolve() != conversion_source:
            raise FileExistsError(f"Existing foundation data came from another source: {destination}")
        return previous
    if destination.exists() and any(destination.glob("*.pt")):
        raise FileExistsError(f"Incomplete PT output already exists: {destination}")
    convert_pickle_to_pt(
        conversion_source, destination, shard_size=shard_size, shuffle=True, seed=seed
    )
    result = json.loads(manifest.read_text(encoding="utf-8"))
    result["success_only"] = bool(success_only)
    result["collection_source"] = str(source)
    result["successful_episodes"] = successful_episode_count
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/data/Getuanhui/avnav_diverse_all_scenes_20261018/episodes")
    parser.add_argument("--destination", default="result_avnav/2026-10-18-avnav-diversity/data/new_train_pt")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.source, args.destination, args.shard_size, args.seed,
                 success_only=not args.include_failures)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
