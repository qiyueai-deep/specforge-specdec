"""One-click PyCharm experiment runner.

Edit only the CONFIGURATION block below, then click PyCharm's green Run button.
Every run creates its own experiments/<EXPERIMENT_NAME>/ directory and does not
overwrite the official artifacts/ directory.
"""

from __future__ import annotations

from pathlib import Path

from specforge.benchmark import format_benchmark, run_benchmark
from specforge.config import TrainConfig
from specforge.correctness import run_correctness_checks
from specforge.plots import generate_plots, show_plot_files
from specforge.report import build_summary, format_summary
from specforge.training import train_project
from specforge.utils import save_json

import time


def _fmt_time(seconds: float) -> str:
    """秒数格式化为 分:秒 或 时:分:秒"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.1f}s"
    return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"


# ============================================================================
# CONFIGURATION: change values here, then click Run in PyCharm.
# ============================================================================

EXPERIMENT_NAME = "T1.5_a85_d400_middledraft"

# ----------------------------------------------------------------------------
# 配置说明：
#   草稿模型 d=96, L=3, H=4 → 约 457K 参数，压缩比 ~11x
#   蒸馏 250 步，α 从 0.5 退火到 0.15，T=1.5
#   前期侧重 KL 分布对齐，后期侧重 CE 任务质量
#   预计总运行时间 ~8 分钟（训练 ~3min + 基准 ~4min + 其他 ~1min）
# ----------------------------------------------------------------------------
FAST_RUN = False

TRAIN_CONFIG = TrainConfig(
    seed=20260820,
    seq_len=288,
    batch_size=24,
    target_steps=300,
    draft_sft_steps=100,
    distill_steps=400,
    learning_rate=0.0025,
    min_learning_rate=0.0008,
    weight_decay=0.01,
    temperature=1.5,           # T²=2.25
    distill_alpha=0.85,         # α 从 0.85 退火至 0.1，Stage 2 纯 CE
    eval_batches=8,
    torch_threads=4,
)
# 快速模式
if FAST_RUN:
    TRAIN_CONFIG.target_steps = 40
    TRAIN_CONFIG.draft_sft_steps = 30
    TRAIN_CONFIG.distill_steps = 80
    TRAIN_CONFIG.eval_batches = 4

CANDIDATE_LENGTHS = (1, 2, 4, 8, 12, 16, 24, 32)
NEW_TOKENS = 64
BENCHMARK_REPEATS = 2
BENCHMARK_WARMUPS = 1
BENCHMARK_THREADS = 1
BENCHMARK_PROMPT_COUNT = 4
SHOW_FIGURES = True   # True = 每次都用系统默认图片查看器打开 3 张图

# ============================================================================


def main() -> None:
    root = Path(__file__).absolute().parent
    experiment_root = root / "experiments" / EXPERIMENT_NAME
    artifact_dir = experiment_root / "artifacts"
    figure_dir = experiment_root / "figures"

    print("=" * 72)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Output:     {experiment_root}")
    print(f"T / alpha / distill steps: {TRAIN_CONFIG.temperature} / "
          f"{TRAIN_CONFIG.distill_alpha} / {TRAIN_CONFIG.distill_steps}")
    print("=" * 72)

    t_total_start = time.perf_counter()
    timings = {}

    print("\n[1/5] Train target, draft SFT, and distilled draft")
    t0 = time.perf_counter()
    train_project(artifact_dir, TRAIN_CONFIG)
    timings["训练"] = time.perf_counter() - t0

    print("\n[2/5] Verify exact greedy decoding and rejection-sampling identity")
    t0 = time.perf_counter()
    correctness = run_correctness_checks(artifact_dir)
    save_json(artifact_dir / "correctness.json", correctness)
    timings["正确性验证"] = time.perf_counter() - t0
    if not correctness["all_greedy_cases_pass"]:
        raise RuntimeError("Greedy exact-equivalence check failed; stop this experiment.")
    if not correctness["rejection_sampling_identity_pass"]:
        raise RuntimeError("Rejection-sampling identity check failed; stop this experiment.")

    print("\n[3/5] Benchmark speculative decoding")
    t0 = time.perf_counter()
    benchmark = run_benchmark(
        artifact_dir=artifact_dir,
        max_new_tokens=NEW_TOKENS,
        repeats=BENCHMARK_REPEATS,
        warmups=BENCHMARK_WARMUPS,
        candidate_lengths=CANDIDATE_LENGTHS,
        benchmark_threads=BENCHMARK_THREADS,
        prompt_count=BENCHMARK_PROMPT_COUNT,
    )
    timings["性能基准"] = time.perf_counter() - t0
    benchmark_path = artifact_dir / "benchmark.json"
    save_json(benchmark_path, benchmark)
    print(format_benchmark(benchmark))

    print("\n[4/5] Print metrics summary")
    t0 = time.perf_counter()
    print(format_summary(build_summary(artifact_dir)))
    timings["汇总报告"] = time.perf_counter() - t0

    print("\n[5/5] Generate experiment figures")
    t0 = time.perf_counter()
    figures = generate_plots(artifact_dir, figure_dir, benchmark_path)
    timings["生成图表"] = time.perf_counter() - t0
    for path in figures:
        print(path.absolute())

    total_time = time.perf_counter() - t_total_start
    timings["总计"] = total_time

    # 打印耗时统计
    print("\n" + "=" * 72)
    print("各阶段耗时")
    print("-" * 72)
    for name, sec in timings.items():
        bar_len = int(sec / max(1, total_time) * 40)
        bar = "█" * bar_len
        print(f"  {name:<10} {_fmt_time(sec):>12}  {bar}")
    print("-" * 72)
    print(f"  总运行时间:  {_fmt_time(total_time)}")
    print("=" * 72)

    if SHOW_FIGURES:
        print(f"\ndisplay mode: {show_plot_files(figures, block=False)}")
    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
