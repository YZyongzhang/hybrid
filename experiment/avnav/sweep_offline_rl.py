"""Run reproducible AVNav CQL/IQL hyperparameter candidates in sequence."""
import argparse
import json
from pathlib import Path

import yaml

from experiment.avnav.config import resolve_path
from experiment.avnav.offline_rl import run as train_offline_rl


CANDIDATES = [
    # Lower discount factors test the shorter effective horizon suggested by
    # AVNav's collection setup, where no extra reward-shaping factors are used.
    dict(name="cql_g0.05_bc3.0_cql1.0", algorithm="cql", gamma=0.05,
         bc_weight=3.0, cql_weight=1.0),
    dict(name="cql_g0.20_bc1.0_cql0.5", algorithm="cql", gamma=0.20,
         bc_weight=1.0, cql_weight=0.5),
    dict(name="cql_g0.20_bc5.0_cql1.0", algorithm="cql", gamma=0.20,
         bc_weight=5.0, cql_weight=1.0),
    dict(name="cql_g0.40_bc1.0_cql0.5", algorithm="cql", gamma=0.40,
         bc_weight=1.0, cql_weight=0.5),
    dict(name="cql_g0.40_bc5.0_cql1.0", algorithm="cql", gamma=0.40,
         bc_weight=5.0, cql_weight=1.0),
    dict(name="cql_g0.20_bc3.0_cql1.0", algorithm="cql", gamma=0.20,
         bc_weight=3.0, cql_weight=1.0),
    dict(name="cql_g0.40_bc3.0_cql1.0", algorithm="cql", gamma=0.40,
         bc_weight=3.0, cql_weight=1.0),
    dict(name="cql_g0.60_bc3.0_cql1.0", algorithm="cql", gamma=0.60,
         bc_weight=3.0, cql_weight=1.0),
    dict(name="cql_g0.60_bc5.0_cql0.5", algorithm="cql", gamma=0.60,
         bc_weight=5.0, cql_weight=0.5),
    # Preserve medium/high horizons as controls: AVNav reward is dense in
    # distance progress, but lower gamma can still overvalue immediate motion.
    dict(name="cql_g0.80_bc3.0_cql0.5", algorithm="cql", gamma=0.80,
         bc_weight=3.0, cql_weight=0.5),
    dict(name="cql_g0.95_bc5.0_cql1.0", algorithm="cql", gamma=0.95,
         bc_weight=5.0, cql_weight=1.0),
    dict(name="iql_g0.20_beta1.0", algorithm="iql", gamma=0.20,
         beta=1.0, expectile=0.7),
    dict(name="iql_g0.40_beta1.0", algorithm="iql", gamma=0.40,
         beta=1.0, expectile=0.7),
    dict(name="iql_g0.80_beta3.0", algorithm="iql", gamma=0.80,
         beta=3.0, expectile=0.7),
    dict(name="iql_g0.95_beta3.0", algorithm="iql", gamma=0.95,
         beta=3.0, expectile=0.7),
]


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sweep(config_path, output_root, epochs=20, updates_per_epoch=200,
          candidate_names=None):
    config_path = Path(config_path).expanduser().resolve()
    output_root = resolve_path(output_root)
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "sweep.json"
    candidates = CANDIDATES
    if candidate_names:
        requested = set(candidate_names)
        known = {row["name"] for row in CANDIDATES}
        missing = requested - known
        if missing:
            raise ValueError(f"Unknown AVNav candidates: {sorted(missing)}")
        candidates = [row for row in CANDIDATES if row["name"] in requested]
    result = {"status": "running", "base_config": str(config_path),
              "foundation_checkpoint": str(resolve_path(base_config["FOUNDATION"]["CHECKPOINT"])),
              "epochs": epochs, "updates_per_epoch": updates_per_epoch,
              "candidate_names": [row["name"] for row in candidates],
              "candidates": []}

    for candidate in candidates:
        output = output_root / candidate["name"]
        summary_path = output / "training_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("training_config", {}).get("foundation_checkpoint") != result["foundation_checkpoint"]:
                raise FileExistsError(f"Candidate belongs to another foundation: {summary_path}")
            result["candidates"].append(dict(candidate, status="reused",
                                              best_epoch=summary["best_epoch"],
                                              best_validation=summary["best_validation"],
                                              output=str(output)))
            _write_json(manifest_path, result)
            continue

        candidate_config = json.loads(json.dumps(base_config))
        rl = candidate_config["OFFLINE_RL"]
        rl["OUTPUT"] = str(output)
        rl["EPOCHS"] = int(epochs)
        rl["UPDATES_PER_EPOCH"] = int(updates_per_epoch)
        rl["ALGORITHM"] = candidate["algorithm"]
        if candidate["algorithm"] == "cql":
            rl["GAMMA"] = candidate["gamma"]
            rl["BC_WEIGHT"] = candidate["bc_weight"]
            rl["CQL_WEIGHT"] = candidate["cql_weight"]
        else:
            rl["GAMMA"] = candidate["gamma"]
            rl["IQL_BETA"] = candidate["beta"]
            rl["EXPECTILE"] = candidate["expectile"]
        output.mkdir(parents=True, exist_ok=True)
        config_file = output / "candidate_config.yaml"
        config_file.write_text(yaml.safe_dump(candidate_config, sort_keys=False), encoding="utf-8")
        print(f"Starting AVNav offline candidate {candidate['name']}", flush=True)
        train_offline_rl(
            str(config_file), gamma_override=candidate["gamma"],
            output_override=str(output),
            bc_weight_override=candidate.get("bc_weight"),
            cql_weight_override=candidate.get("cql_weight"),
            algorithm_override=candidate["algorithm"],
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["candidates"].append(dict(candidate, status="completed",
                                          best_epoch=summary["best_epoch"],
                                          best_validation=summary["best_validation"],
                                          train_samples=summary["train_samples"],
                                          output=str(output)))
        _write_json(manifest_path, result)

    result["status"] = "completed"
    _write_json(manifest_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment/avnav/config/offline_rl_diverse_foundation_avnav_20261018.yaml")
    parser.add_argument("--output-root", default="result_avnav/2026-10-18-avnav-diversity/offline/candidates")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--updates-per-epoch", type=int, default=200)
    parser.add_argument("--candidate-names", nargs="+", default=None,
                        help="Optional subset of candidate names")
    args = parser.parse_args(argv)
    print(json.dumps(sweep(args.config, args.output_root, args.epochs,
                           args.updates_per_epoch, args.candidate_names),
              indent=2), flush=True)


if __name__ == "__main__":
    main()
