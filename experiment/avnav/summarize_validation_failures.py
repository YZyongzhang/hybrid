"""Summarize raw-policy AVNav failure and collision patterns on validation only."""
import argparse
import json
from pathlib import Path


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _summarize(rows):
    boundaries = (("0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9),
                  ("10-19", 10, 19), ("20+", 20, None))
    bins = []
    for name, low, high in boundaries:
        group = [row for row in rows if row.get("collisions", 0) >= low and
                 (high is None or row.get("collisions", 0) <= high)]
        if not group:
            continue
        bins.append({
            "collision_bin": name,
            "episodes": len(group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "spl": sum(float(row["spl"]) for row in group) / len(group),
            "mean_steps": sum(float(row["steps"]) for row in group) / len(group),
            "timeout_rate": sum(int(row["steps"]) >= 200 for row in group) / len(group),
        })
    return {
        "episodes": len(rows),
        "errors": sum(row.get("error") is not None for row in rows),
        "mean_collisions": sum(row.get("collisions", 0) for row in rows) / max(1, len(rows)),
        "bins": bins,
    }


def summarize(run_dir, output_path):
    run_dir = Path(run_dir).expanduser().resolve()
    report = {
        "source": str(run_dir),
        "validation_only": True,
        "action_handling": "raw actor actions; collision counts are diagnostic only",
        "splits": {},
    }
    for split in ("val_multiple", "val_multiple_unheard"):
        rows = _read(run_dir / f"{split}_episodes.jsonl")
        report["splits"][split] = _summarize(rows)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Final validation directory containing both episode JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.output), indent=2,
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
