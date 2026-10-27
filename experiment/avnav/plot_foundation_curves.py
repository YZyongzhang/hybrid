"""Plot AVNav staged-foundation direction and action metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plot(foundation_dir, output_path=None):
    foundation_dir = Path(foundation_dir).expanduser().resolve()
    audio = _read_jsonl(foundation_dir / "audio_stage/metrics.jsonl")
    action = _read_jsonl(foundation_dir / "action_stage/metrics.jsonl")
    if not audio or not action:
        raise ValueError("Audio and action stage metrics are both required")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    epochs = [int(row["epoch"]) for row in audio]
    ax = axes[0, 0]
    ax.plot(epochs, [row["train"]["angle_loss"] for row in audio], label="train")
    ax.plot(epochs, [row["val_heard"]["angle_loss"] for row in audio], label="heard val")
    ax.plot(epochs, [row["val_unheard"]["angle_loss"] for row in audio], label="unheard val")
    ax.set_title("Audio source-direction cross entropy")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [row["train"]["angle_accuracy"] for row in audio], label="train")
    ax.plot(epochs, [row["val_heard"]["angle_accuracy"] for row in audio], label="heard val")
    ax.plot(epochs, [row["val_unheard"]["angle_accuracy"] for row in audio], label="unheard val")
    ax.set_title("Audio direction accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    epochs = [int(row["epoch"]) for row in action]
    ax = axes[1, 0]
    ax.plot(epochs, [row["train"]["action_loss"] for row in action], label="train")
    ax.plot(epochs, [row["val_heard"]["action_loss"] for row in action], label="heard val")
    ax.plot(epochs, [row["val_unheard"]["action_loss"] for row in action], label="unheard val")
    ax.set_title("Greedy expert action cross entropy")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, [row["train"]["action_accuracy"] for row in action], label="train")
    ax.plot(epochs, [row["val_heard"]["action_accuracy"] for row in action], label="heard val")
    ax.plot(epochs, [row["val_unheard"]["action_accuracy"] for row in action], label="unheard val")
    ax.set_title("Greedy expert action accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("AVNav greedy-only staged foundation", fontsize=15)
    output_path = Path(output_path).expanduser().resolve() if output_path else \
        foundation_dir / "foundation_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(plot(args.foundation_dir, args.output or None))


if __name__ == "__main__":
    main()
