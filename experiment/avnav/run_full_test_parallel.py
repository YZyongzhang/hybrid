"""Run complete AVNav test splits in resumable parallel episode shards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.avnav.aggregate_online_shards import aggregate


def run(config, checkpoint, output, splits, episodes=1000, workers=4,
        max_concurrent=None):
    config = Path(config).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    shard_root = output / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(episodes, start + (episodes + workers - 1) // workers))
              for start in range(0, episodes, (episodes + workers - 1) // workers)]
    max_concurrent = workers if max_concurrent is None else int(max_concurrent)
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be positive")
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "1"

    for split in splits:
        for offset in range(0, len(ranges), max_concurrent):
            running = []
            for start, end in ranges[offset:offset + max_concurrent]:
                shard = shard_root / f"{split}_{start}_{end}"
                shard.mkdir(parents=True, exist_ok=True)
                metrics_path = shard / f"{split}.json"
                episodes_path = shard / f"{split}_episodes.jsonl"
                if metrics_path.is_file() and episodes_path.is_file():
                    import json
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics.get("episodes") == end - start:
                        continue
                command = [sys.executable, "-m", "experiment.avnav.evaluate_online",
                           "--config", str(config), "--checkpoint", str(checkpoint),
                           "--output", str(shard), "--episodes", str(episodes),
                           "--splits", split, "--index-range", str(start), str(end)]
                log = (shard / "evaluation.log").open("a", encoding="utf-8")
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                running.append((process, log, shard))
            failed = []
            for process, log, shard in running:
                code = process.wait()
                log.close()
                if code:
                    _, start, end = shard.name.rsplit("_", 2)
                    metrics_path = shard / f"{split}.json"
                    episodes_path = shard / f"{split}_episodes.jsonl"
                    complete = False
                    if metrics_path.is_file() and episodes_path.is_file():
                        try:
                            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                            rows = [line for line in episodes_path.read_text(
                                encoding="utf-8").splitlines() if line]
                            complete = (metrics.get("episodes") == int(end) - int(start) and
                                        len(rows) == int(end) - int(start))
                        except (OSError, ValueError, TypeError):
                            complete = False
                    if complete:
                        print(f"{shard}: exit {code} after complete shard outputs; "
                              "keeping the validated episode records", flush=True)
                    else:
                        failed.append((str(shard), code))
            if failed:
                raise RuntimeError(f"Failed AVNav evaluation shards: {failed}")

    final_root = output / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    return aggregate(final_root, shard_root, episodes, tuple(splits))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["test_multiple", "test_multiple_unheard"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Maximum live simulation workers (defaults to --workers)")
    args = parser.parse_args(argv)
    result = run(args.config, args.checkpoint, args.output, args.splits,
                 args.episodes, args.workers, args.max_concurrent)
    import json
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
