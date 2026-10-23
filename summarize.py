from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from lander import VARIANTS
from lander import results
from lander.config import RESULTS_DIR, SOLVED_SCORE

LABELS = {
    "baseline": "Baseline",
    "time_penalty": "Time penalty",
    "hover_hack": "Hover hack",
    "hover_fixed": "Hover fixed",
}
COLORS = {
    "baseline": "#2a78d6",
    "time_penalty": "#eb6834",
    "hover_hack": "#1baf7a",
    "hover_fixed": "#c98500",
}


def markdown_table(runs):
    lines = [
        "| Reward | Seed | Score (mean ± std) | Landed | Crashed | Timed out | Train (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if "eval_mean" not in run:
            continue
        n = run.get("eval_episodes", 20)
        lines.append(
            f"| {LABELS[run['variant']]} | {run['seed']} | "
            f"{run['eval_mean']:.1f} ± {run['eval_std']:.1f} | "
            f"{run['landed']}/{n} | {run['crashed']}/{n} | "
            f"{run['timed_out']}/{n} | {run.get('train_minutes', 0):.1f} |"
        )
    return "\n".join(lines)


def plot_curves(runs, index, ylabel, filename, solved_line=False):
    by_variant = defaultdict(list)
    for run in runs:
        if run.get("curve"):
            by_variant[run["variant"]].append(run)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for variant in VARIANTS:
        for i, run in enumerate(sorted(by_variant.get(variant, []), key=lambda r: r["seed"])):
            steps = [p[0] for p in run["curve"]]
            values = [p[index] for p in run["curve"]]
            ax.plot(steps, values, color=COLORS[variant], linestyle="-" if i == 0 else "--",
                    linewidth=1.6, label=LABELS[variant] if i == 0 else None)
    if solved_line:
        ax.axhline(SOLVED_SCORE, color="gray", linestyle=":", linewidth=1)
        ax.text(0.005, SOLVED_SCORE, "solved", transform=ax.get_yaxis_transform(),
                va="bottom", color="gray", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x else "0"))
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    runs = sorted(results.load_all(), key=lambda r: (list(VARIANTS).index(r["variant"]), r["seed"]))
    if not runs:
        print("No results found. Run train.py and evaluate.py first.")
        return
    print(markdown_table(runs))
    params = {r.get("parameters") for r in runs if r.get("parameters")}
    if params:
        print(f"\nModel parameters: {', '.join(f'{p:,}' for p in sorted(params))}")
    print(plot_curves(runs, 1, "Mean episode score (shaped reward)", "learning_curves.png",
                      solved_line=True))
    print(plot_curves(runs, 2, "Mean episode length (frames)", "episode_length.png"))


if __name__ == "__main__":
    main()
