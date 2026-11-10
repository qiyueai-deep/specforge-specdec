"""One-click PyCharm experiment runner (DRAFT-ONLY preset: only the draft shrunk).

This is a COPY that shrinks ONLY the DRAFT model architecture (~0.75x width)
while keeping the TARGET at its original large size. Speculative decoding saves
on the expensive target model, so keeping the target big preserves the ~2x
speedup headline, while the tiny draft keeps draft-side cost/training low.

IMPORTANT: this file does NOT modify any source under specforge/. It only
monkeypatches specforge.training.draft_config at runtime (inside main) to return
a smaller ModelConfig. The on-disk specforge/ modules stay byte-identical, so
your original project is untouched.

Edit only the CONFIGURATION block below, then click PyCharm's green Run button.
Every run creates its own experiments/<EXPERIMENT_NAME>/ directory.
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


# ----------------------------------------------------------------------------
# Model scaling: ONLY the DRAFT is shrunk (~0.75x width). The TARGET keeps its
# original large architecture, so the "expensive target" that speculative
# decoding saves on is preserved -> the ~2x speedup headline is kept, while the
# small draft trains fast.
#   target: ORIGINAL (d 256, L 6, H 8, FF 1024) -> ~5.28M  (unchanged)
#   draft : d  96 -> 72, H 4->3 (head_dim 24), FF  384->288 -> ~334K
# ----------------------------------------------------------------------------
import specforge.training as _tr

_ORIG_DRAFT_CFG = _tr.draft_config


def _scaled_draft_config(vocab_size: int):
    c = _ORIG_DRAFT_CFG(vocab_size)
    c.d_model = 72
    c.n_layers = 3
    c.n_heads = 3      # head_dim = 72/3 = 24 (same as original 96/4)
    c.d_ff = 288       # 4 * d_model
    return c


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

EXPERIMENT_NAME = "T2.0_a85_draftonly"

# ----------------------------------------------------------------------------
# 又快又好档（target 质量满分 + 训练多线程加速）：
#   目标 d=256 不变（昂贵，投机解码省大头），target_steps=220 -> val CE ~0.13 质量满分（推理不打折）
#   草稿 d=96->72（便宜），d_model 不一致 -> 走 SFT+蒸馏对齐
#   torch_threads=8：训练阶段多线程加速（训练耗时降；推理基准仍单线程以保加速比数字）
#   基准生成 96、重复 3 次；候选 K 到 48，Best_K 能选大
#   预期：val CE ~0.13（文字连贯）+ 加速比 ~1.9-2.0x + 总耗时 ~8-9 分钟（远快于原 20 分钟）
# ----------------------------------------------------------------------------
FAST_RUN = False

TRAIN_CONFIG = TrainConfig(
    seed=20260820,
    seq_len=288,               # 必须 >= prompt(18) + NEW_TOKENS(96)，114 < 288 安全（分布内，Best_K 不塌）
    batch_size=24,
    target_steps=220,          # 质量满分：与原完整档一致，val CE ~0.13，推理质量不打折（target 是系统天花板）
    draft_sft_steps=100,       # 草稿从头 SFT（d_model 不一致，不走层拷贝），100 步足够
    distill_steps=150,         # 草稿蒸馏对齐，150 步已足够稳
    learning_rate=0.0025,
    min_learning_rate=0.00025,
    weight_decay=0.01,
    temperature=2.0,
    distill_alpha=0.85,         # 侧重 KL
    eval_batches=6,            # 训练期评估报告，不改模型
    torch_threads=8,           # 训练多线程加速（推理基准仍单线程以保加速比数字）
)
# 快速冒烟模式（仅验证管道能跑通，目标损失会很高、文本为乱码，勿用于正式成果）
if FAST_RUN:
    TRAIN_CONFIG.target_steps = 40
    TRAIN_CONFIG.draft_sft_steps = 30
    TRAIN_CONFIG.distill_steps = 80
    TRAIN_CONFIG.eval_batches = 4

CANDIDATE_LENGTHS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48)  # 元组（非生成器）；K 到 48 保证 Best_K 能选大
NEW_TOKENS = 96   # 基准生成长度；18+96=114 < 288 安全，分布内
BENCHMARK_REPEATS = 3  # 重复 3 次估算加速比更稳定
BENCHMARK_THREADS = 1
SHOW_FIGURES = True   # True = 每次用系统默认图片查看器打开 3 张图

# ============================================================================


def main() -> None:
    # 运行时只把【草稿】模型配置替换为缩小版（仅本次进程，不改动磁盘上的 specforge/ 源码）
    # 目标模型保持原始大小（d=256），以保证投机解码的加速比招牌
    _tr.draft_config = _scaled_draft_config

    root = Path(__file__).absolute().parent
    experiment_root = root / "experiments" / EXPERIMENT_NAME
    artifact_dir = experiment_root / "artifacts"
    figure_dir = experiment_root / "figures"

    print("=" * 72)
    print(f"Experiment: {EXPERIMENT_NAME}  (draft model shrunk 0.75x, target kept original)")
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
