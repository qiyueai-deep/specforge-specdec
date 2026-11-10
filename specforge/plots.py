from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .utils import load_json


COLORS = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "purple": "#7c3aed",
    "red": "#dc2626",
    "gray": "#64748b",
}


def _save_figure(fig, output: Path, filename: str) -> Path:
    """Save a figure, falling back to a unique filename if Windows locks it."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    try:
        fig.savefig(path, bbox_inches="tight")
        return path
    except PermissionError:
        stem, suffix = path.stem, path.suffix
        for index in range(2, 1000):
            fallback = output / f"{stem}_run{index}{suffix}"
            try:
                fig.savefig(fallback, bbox_inches="tight")
                print(f"[plot] 文件被占用，已改存为: {fallback.name}")
                return fallback
            except PermissionError:
                continue
        raise


def _ema(values: list[float], decay: float = 0.55) -> list[float]:
    """Exponential moving average for visualizing noisy batch losses."""
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(decay * smoothed[-1] + (1.0 - decay) * float(value))
    return smoothed


def _axis_ticks(total_steps: int, max_ticks: int = 10) -> list[int]:
    """Evenly spaced, round integer ticks derived from the real run length.

    The plot used to hardcode the 220-step schedule, which pushed every tick
    outside the visible range for the quick smoke-test schedules.
    """
    if total_steps <= 0:
        return [0]
    raw_step = max(1, math.ceil(total_steps / max_ticks))
    step = max(5, math.ceil(raw_step / 5) * 5)
    ticks = list(range(0, total_steps + 1, step))
    if ticks[-1] != total_steps:
        ticks.append(total_steps)
    return ticks


def _running_min(values: list[float]) -> list[float]:
    """Best validation value observed up to each checkpoint."""
    best = float("inf")
    result: list[float] = []
    for value in values:
        best = min(best, float(value))
        result.append(best)
    return result


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_training(manifest: dict, output: Path) -> Path:
    _style()
    history = manifest["history"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    target = history["target"]
    draft = history["draft_sft"]
    distill = history["distillation"]
    # When layer copy-initialisation is used, the draft has no SFT curve: the
    # single recorded point is the post-copy KL, not a cross-entropy.
    draft_label = "Draft fixed-validation CE" if len(draft) > 1 else "Draft after copy-init (KL)"
    for rows, label, color, marker in (
        (target, "Target fixed-validation CE", COLORS["blue"], "o"),
        (draft, draft_label, COLORS["orange"], "s"),
    ):
        steps = [row["step"] for row in rows]
        values = [row["loss"] for row in rows]
        axes[0].scatter(steps, values, marker=marker, color=color, s=18, alpha=0.22)
        axes[0].plot(steps, values, marker + "-", color=color, linewidth=2.0, label=label)
    axes[0].set(
        title="Supervised training",
        xlabel="Optimization step",
        ylabel="Fixed-validation CE",
        ylim=(0.0, 5.5),
    )
    axes[0].set_yticks([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[0].legend(frameon=False)
    baseline = distill[0]
    trained_steps = [row for row in distill if row["step"] > 0]
    for key, label, color, marker in (
        ("loss", "Total", COLORS["purple"], "o"),
        ("kd", "KL component", COLORS["red"], "s"),
        ("ce", "CE component", COLORS["green"], "^")
    ):
        steps = [row["step"] for row in trained_steps]
        values = [row[key] for row in trained_steps]
        if not values:
            continue
        trend = _ema(values, decay=0.35)
        axes[1].scatter([baseline["step"]], [baseline[key]], marker=marker, color=color,
                        s=18, alpha=0.16)
        axes[1].scatter(steps, values, marker=marker, color=color,
                        s=13, alpha=0.13)
        axes[1].plot(
            steps, trend, "-", color=color, linewidth=2.1,
            label=f"{label} (final={values[-1]:.4f})",
        )
    loss_values = [float(row[key]) for row in distill for key in ("loss", "kd", "ce")]
    axis_top = max(1.5, max(loss_values) * 1.08)
    total_distill_steps = max((int(row["step"]) for row in distill), default=0)
    axes[1].set(
        title="Knowledge distillation loss",
        xlabel="Distillation step",
        ylabel="Validation loss",
        # Linear scale: 0.1 is exactly one tenth of the distance from 0 to 1.
        ylim=(0.0, axis_top),
        xlim=(0, max(total_distill_steps, 1)),
    )
    axes[1].set_xticks(_axis_ticks(total_distill_steps))
    axes[1].set_yticks([0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 3.5])
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    metrics = manifest["metrics"]
    stages = ["Initial", "After SFT", "After KD"]
    agreement = [
        metrics["draft_before_sft"]["token_top1_agreement"],
        metrics["draft_after_sft"]["token_top1_agreement"],
        metrics["draft_after_distillation"]["token_top1_agreement"],
    ]
    axes[2].bar(stages, agreement, color=[COLORS["gray"], COLORS["orange"], COLORS["green"]], width=0.62)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set(title="Draft-target alignment", xlabel="Training stage", ylabel="Token top-1 agreement", ylim=(0, 1.05))
    for index, value in enumerate(agreement):
        axes[2].text(index, value + 0.025, f"{value:.2%}", ha="center")
    fig.suptitle("SpecForge training and alignment curves", fontsize=14)
    fig.tight_layout()
    path = _save_figure(fig, output, "training_curves.png")
    plt.close(fig)
    return path


def plot_benchmark(benchmark: dict, output: Path) -> Path:
    _style()
    rows = benchmark["speculative"]
    k = [row["draft_k"] for row in rows]
    acceptance = [row["mean_acceptance_rate"] for row in rows]
    reduction = [row["target_call_reduction"] for row in rows]
    throughput = [row["tokens_per_second"] for row in rows]
    speedup = [row["wall_clock_speedup"] for row in rows]
    baseline = benchmark["baseline"]["tokens_per_second"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    axes[0].plot(k, acceptance, "o-", color=COLORS["blue"], label="Acceptance rate")
    axes[0].plot(k, reduction, "s--", color=COLORS["orange"], label="Target-call reduction")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set(title="Algorithmic efficiency", xlabel="Draft length K", ylabel="Rate", xticks=k, ylim=(-0.08, 1.05))
    axes[0].legend(frameon=False)
    axes[1].plot(k, throughput, "o-", color=COLORS["green"], linewidth=2)
    axes[1].axhline(baseline, color=COLORS["gray"], linestyle="--", label=f"AR baseline: {baseline:.1f}")
    axes[1].set(title="CPU throughput", xlabel="Draft length K", ylabel="Generated token/s", xticks=k,
                ylim=(0, max(throughput + [baseline]) * 1.12))
    axes[1].legend(frameon=False)
    axes[2].plot(k, speedup, "o-", color=COLORS["purple"], linewidth=2)
    axes[2].axhline(1.0, color=COLORS["gray"], linestyle="--", label="Break-even")
    axes[2].set(title="End-to-end speedup", xlabel="Draft length K", ylabel="Speedup vs autoregressive", xticks=k,
                ylim=(0, max(speedup + [1.0]) * 1.12))
    axes[2].legend(frameon=False)
    best_index = max(range(len(speedup)), key=speedup.__getitem__)
    axes[2].annotate(
        f"best {speedup[best_index]:.3f}x",
        xy=(k[best_index], speedup[best_index]), xytext=(-88, -34),
        textcoords="offset points", arrowprops={"arrowstyle": "->", "color": COLORS["gray"]},
    )
    fig.suptitle("Speculative decoding K ablation", fontsize=14)
    fig.tight_layout()
    path = _save_figure(fig, output, "speculative_benchmark.png")
    plt.close(fig)
    return path


def plot_model_comparison(manifest: dict, output: Path) -> Path:
    _style()
    target = manifest["target_parameters"]
    draft = manifest["draft_parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))

    # Left: model size with proper Y-axis headroom
    sizes = [target / 1e6, draft / 1e6]
    bars = axes[0].bar(["Target", "Draft"], sizes, color=[COLORS["blue"], COLORS["orange"]], width=0.58)
    axes[0].set(title="Model size", ylabel="Parameters (million)", ylim=(0, max(sizes) * 1.45))
    axes[0].set_xticks([0, 1])
    for bar, val in zip(bars, (target, draft)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val / 1e6 + max(sizes) * 0.02,
                     f"{val / 1e6:.3f}M", ha="center", va="bottom", fontweight="bold")

    # Right: KL divergence on log scale
    metrics = manifest["metrics"]
    stages = ["Before SFT", "After SFT", "After KD"]
    kl = [
        metrics["draft_before_sft"]["kl_target_to_draft"],
        metrics["draft_after_sft"]["kl_target_to_draft"],
        metrics["draft_after_distillation"]["kl_target_to_draft"],
    ]
    kl_colors = [COLORS["gray"], COLORS["orange"], COLORS["green"]]
    bars2 = axes[1].bar(stages, kl, color=kl_colors, width=0.6)
    axes[1].set(title="Distribution alignment", ylabel="KL(target || draft)", yscale="log")
    axes[1].set_ylim(min(kl) * 0.5, max(kl) * 2.5)
    for bar, val in zip(bars2, kl):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    fig.suptitle(f"Draft compression ratio: {manifest['parameter_compression_ratio']:.2f}x", fontsize=14)
    fig.tight_layout()
    path = _save_figure(fig, output, "model_alignment.png")
    plt.close(fig)
    return path


def generate_plots(
    artifact_dir: str | Path = "artifacts",
    output_dir: str | Path = "figures",
    benchmark_path: str | Path | None = None,
) -> list[Path]:
    artifact_dir, output = Path(artifact_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_json(artifact_dir / "manifest.json")
    benchmark = load_json(benchmark_path or artifact_dir / "benchmark.json")
    return [
        plot_training(manifest, output),
        plot_benchmark(benchmark, output),
        plot_model_comparison(manifest, output),
    ]


def show_plot_files(paths: list[str | Path], block: bool = True) -> str:
    """Launch a clean viewer process so plotting never breaks headless saving."""
    if os.environ.get("SPECFORGE_NO_PLOT_POPUP") == "1":
        return "suppressed"
    paths = [Path(path).absolute() for path in paths]
    command = [sys.executable, "-m", "specforge.viewer", *[str(path) for path in paths]]
    environment = os.environ.copy()
    environment.pop("MPLBACKEND", None)
    if block:
        completed = subprocess.run(command, env=environment, check=False)
        return f"viewer-process:{completed.returncode}"
    subprocess.Popen(command, env=environment)
    return "viewer-process:background"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready plots from saved JSON artifacts")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--benchmark", help="Optional alternative benchmark JSON")
    parser.add_argument("--show", action="store_true", help="Open generated plots after saving")
    args = parser.parse_args()
    paths = generate_plots(args.artifact_dir, args.output_dir, args.benchmark)
    for path in paths:
        print(path)
    if args.show:
        print(f"display mode: {show_plot_files(paths)}")


if __name__ == "__main__":
    main()
