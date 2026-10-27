"""Plot discrete SAC-CQL training and validation metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot(metrics_path, output_path=None):
    metrics_path = Path(metrics_path).expanduser().resolve()
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"No offline-RL metrics in {metrics_path}")
    epochs = [int(row["epoch"]) for row in rows]

    def series(key):
        return [float(row[key]) for row in rows]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(epochs, series("critic_loss"), label="critic loss")
    ax.plot(epochs, series("td_loss"), label="TD loss")
    ax.plot(epochs, series("cql_gap"), label="CQL gap")
    ax.set_title("Critic")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, series("actor_loss"), label="actor loss")
    ax.plot(epochs, series("sac_loss"), label="SAC objective")
    ax.plot(epochs, series("bc_loss"), label="behavior cloning loss")
    ax.set_title("Actor objectives")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, series("policy_entropy"), color="tab:purple", label="policy entropy")
    ax.set_title("Policy entropy (nats)")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, series("q_mean"), label="mean min Q")
    ax.plot(epochs, series("target_mean"), label="mean TD target")
    ax.set_title("Value scale")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 1]
    heard = [float(row["validation"]["splits"]["validation"]["action_nll"])
             for row in rows]
    unheard = [float(row["validation"]["splits"]["validation_unheard"]["action_nll"])
               for row in rows]
    ax.plot(epochs, heard, label="heard validation NLL")
    ax.plot(epochs, unheard, label="unheard validation NLL")
    ax.set_title("Held-out action NLL")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(epochs, series("sample_reward_mean"), color="tab:green", label="sample reward mean")
    ax.set_title("Sampled training batch")
    ax.set_xlabel("Epoch")
    ax.legend()
    terminal_ax = ax.twinx()
    terminal_ax.plot(epochs, series("sample_terminal_fraction"), color="tab:orange",
                     linestyle="--", label="terminal fraction")
    terminal_ax.set_ylabel("Terminal fraction")
    terminal_ax.legend(loc="lower right")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle("AVNav discrete SAC-CQL training curves", fontsize=15)
    output_path = Path(output_path).expanduser().resolve() if output_path else metrics_path.with_name(
        "training_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    print(plot(args.metrics, args.output or None))


if __name__ == "__main__":
    main()
