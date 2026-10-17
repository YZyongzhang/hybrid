"""Retrain the frozen-feature linear AVNav action probe with balanced classes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment.avnav.offline_rl import FeatureDataset
from experiment.avnav.evaluate_linear_probe import _metric_row


ACTION_NAMES = ("STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT")


def _balanced_epoch_indices(actions, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    target = len(actions) // 4
    selections = []
    counts = []
    for action in range(4):
        indices = torch.nonzero(actions == action, as_tuple=False).flatten()
        counts.append(len(indices))
        if len(indices) >= target:
            selected = indices[torch.randperm(len(indices), generator=generator)[:target]]
        else:
            selected = indices[torch.randint(len(indices), (target,), generator=generator)]
        selections.append(selected)
    balanced = torch.cat(selections)
    return balanced[torch.randperm(len(balanced), generator=generator)], counts, target


def _predict(model, features, device, batch_size):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            outputs.append(model(features[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(outputs)


def _per_action(logits, actions):
    prediction = logits.argmax(dim=1)
    rows = {}
    for action, name in enumerate(ACTION_NAMES):
        mask = actions == action
        predicted_as_action = prediction == action
        true_positive = int((mask & predicted_as_action).sum())
        support = int(mask.sum())
        predicted_count = int(predicted_as_action.sum())
        rows[name] = {
            "action_id": action,
            "support": support,
            "recall": true_positive / max(support, 1),
            "precision": true_positive / max(predicted_count, 1),
            "f1": 2 * true_positive / max(support + predicted_count, 1),
        }
    return rows


def _evaluate(model, data, device, batch_size):
    logits = _predict(model, data["features"], device, batch_size)
    return {
        "overall": _metric_row(logits, data["actions"]),
        "per_action": _per_action(logits, data["actions"]),
        "logits": logits,
    }


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    train_data = FeatureDataset(args.train_features).data
    train_features = train_data["states"].float()
    train_actions = train_data["actions"].long()
    class_counts = [int((train_actions == action).sum()) for action in range(4)]
    if any(count == 0 for count in class_counts):
        raise ValueError(f"All four action classes must occur in training: {class_counts}")

    probe = nn.Linear(train_features.shape[1], 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    inputs = train_features.to(device)
    targets = train_actions.to(device)

    validation_data = {}
    for split, path in (("heard", args.heard_features), ("unheard", args.unheard_features)):
        data = torch.load(path, map_location="cpu", weights_only=False)
        validation_data[split] = {
            "features": data["features"].float(),
            "actions": data["actions"].long(),
            "scenes": data["scenes"],
            "sounds": data["sounds"],
        }

    # The original-distribution linear probe is the paired baseline.
    baseline = nn.Linear(train_features.shape[1], 4)
    baseline.load_state_dict(torch.load(args.baseline_probe, map_location="cpu", weights_only=True))
    baseline = baseline.to(device).eval()
    baseline_metrics = {}
    for split, data in validation_data.items():
        baseline_metrics[split] = _evaluate(baseline, data, device, args.eval_batch_size)

    epoch_history = []
    for epoch in range(1, args.epochs + 1):
        probe.train()
        indices, source_counts, target_per_action = _balanced_epoch_indices(
            train_actions, args.seed + epoch
        )
        running_loss = 0.0
        seen = 0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size].to(device)
            logits = probe(inputs[batch_indices])
            loss = F.cross_entropy(logits, targets[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch_indices)
            running_loss += float(loss.detach()) * count
            seen += count

        row = {
            "epoch": epoch,
            "balanced_train": {
                "samples": seen,
                "source_class_counts": source_counts,
                "sampled_class_counts": [target_per_action] * 4,
                "loss": running_loss / max(seen, 1),
            },
            "validation": {},
        }
        for split, data in validation_data.items():
            evaluated = _evaluate(probe, data, device, args.eval_batch_size)
            row["validation"][split] = {
                "overall": evaluated["overall"],
                "per_action": evaluated["per_action"],
            }
        epoch_history.append(row)
        if epoch in (1, args.epochs) or epoch % 5 == 0:
            print(json.dumps({"epoch": epoch,
                              "heard": row["validation"]["heard"]["per_action"],
                              "unheard": row["validation"]["unheard"]["per_action"]}),
                  flush=True)

    final = epoch_history[-1]
    report = {
        "protocol": {
            "train_features": str(Path(args.train_features).resolve()),
            "frozen_feature_dim": int(train_features.shape[1]),
            "class_names": list(ACTION_NAMES),
            "source_class_counts": class_counts,
            "balanced_samples_per_class_per_epoch": len(train_actions) // 4,
            "oversampling_with_replacement": [
                class_counts[action] < len(train_actions) // 4 for action in range(4)
            ],
            "downsampling_without_replacement": [
                class_counts[action] > len(train_actions) // 4 for action in range(4)
            ],
            "epochs": args.epochs,
            "validation_used_for_checkpoint_selection": False,
            "baseline": "previous 40-epoch linear probe trained on the natural action distribution",
        },
        "baseline_natural_distribution": {},
        "balanced_distribution": {},
        "epochs": epoch_history,
    }
    for split, data in validation_data.items():
        base = baseline_metrics[split]
        balanced = _evaluate(probe, data, device, args.eval_batch_size)
        report["baseline_natural_distribution"][split] = {
            "overall": base["overall"], "per_action": base["per_action"]
        }
        report["balanced_distribution"][split] = {
            "overall": balanced["overall"], "per_action": balanced["per_action"]
        }
        groups = {}
        for group_type, values in (("scene", data["scenes"]), ("sound", data["sounds"])):
            group_rows = {}
            for group in sorted(set(values)):
                mask = torch.tensor([value == group for value in values])
                group_rows[group] = {
                    "baseline": _per_action(base["logits"][mask], data["actions"][mask]),
                    "balanced": _per_action(balanced["logits"][mask], data["actions"][mask]),
                }
            groups[group_type] = group_rows
        report["balanced_distribution"][split]["by_group"] = groups

    torch.save(probe.cpu().state_dict(), output / "balanced_linear_probe.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "class_counts": class_counts,
        "baseline": report["baseline_natural_distribution"],
        "balanced": report["balanced_distribution"],
    }, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", default="result/2026-10-13-avnav-staged-grl-gamma/offline/shared_features/e7645d17895e8686/train")
    parser.add_argument("--heard-features", default="result/2026-10-14-avnav-linear-probe/heard_features.pt")
    parser.add_argument("--unheard-features", default="result/2026-10-14-avnav-linear-probe/unheard_features.pt")
    parser.add_argument("--baseline-probe", default="result/2026-10-14-avnav-linear-probe/linear_probe.pt")
    parser.add_argument("--output", default="result/2026-10-14-avnav-balanced-action-probe")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
