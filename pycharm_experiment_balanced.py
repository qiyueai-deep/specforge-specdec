"""One-click PyCharm experiment runner (BALANCED preset).

This is a COPY of pycharm_experiment.py with the "lean" (fast) parameter set:
shortest runtime (~5-6 min) while keeping coherent text and a not-too-small Best_K.
The original pycharm_experiment.py is left unchanged.

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

EXPERIMENT_NAME = "T2.0_a85_lean"

# ----------------------------------------------------------------------------
# 再压一档配置说明（追求最短运行时间，约 5~6 分钟）：
#   草稿模型 d=160, L=2, H=4 → 约 450K 参数，压缩比 ~11x
#   训练步数进一步压低（target 150→80, distill 250→150, sft 80→40, eval 6→4）
#   基准生成长度 96、重复 2 次（更快）
#   候选 K 仍保留到 48，保证 Best_K 能选到较大值
#   注意：生成较短，Best_K 可能落在 12~16 区间，文本仍连贯；加速比曲线略糙
# ----------------------------------------------------------------------------
FAST_RUN = False

TRAIN_CONFIG = TrainConfig(
    seed=20260820,
    seq_len=288,               # 必须 >= prompt(18) + NEW_TOKENS(96)，114 < 288 安全；否则生成落分布外、接受率崩、Best_K 塌到 1
    batch_size=24,
    target_steps=80,           # 再压一档：150→80，小语料已充分收敛，文本仍连贯
    draft_sft_steps=40,        # 再压一档：80→40，维度一致时草稿靠拷贝前几层初始化，SFT 影响小
    distill_steps=150,         # 再压一档：250→150，对齐 150 步已足够稳
    learning_rate=0.0025,
    min_learning_rate=0.00025,
    weight_decay=0.01,
    temperature=2.0,
    distill_alpha=0.85,         # 侧重 KL
    eval_batches=4,            # 再压一档：6→4，仅影响训练期报告，不改模型
    torch_threads=4,
)
# 快速冒烟模式（仅验证管道能跑通，目标损失会很高、文本为乱码，勿用于正式成果）
if FAST_RUN:
    TRAIN_CONFIG.target_steps = 40
    TRAIN_CONFIG.draft_sft_steps = 30
    TRAIN_CONFIG.distill_steps = 80
    TRAIN_CONFIG.eval_batches = 4

CANDIDATE_LENGTHS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
NEW_TOKENS = 96   # 再压一档：160→96，基准约快 1.6x，且 18+96=114 < 288 安全
BENCHMARK_REPEATS = 2  # 再压一档：3→2，基准更快；2 次重复足够估算加速比
BENCHMARK_THREADS = 1
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
        warmups=2,
        candidate_lengths=CANDIDATE_LENGTHS,
        benchmark_threads=BENCHMARK_THREADS,
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
