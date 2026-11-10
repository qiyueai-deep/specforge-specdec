"""Fast from-scratch PyCharm training preset (CPU, target < 180 seconds)."""

from __future__ import annotations

import time
from pathlib import Path

from specforge.benchmark import format_benchmark, run_benchmark
from specforge.config import ModelConfig, TrainConfig
from specforge.correctness import run_correctness_checks
from specforge.plots import generate_plots, show_plot_files
from specforge.report import build_summary, format_summary
from specforge.training import train_project
from specforge.utils import save_json

import specforge.training as training


# =============================== CONFIGURATION ==============================
EXPERIMENT_NAME = "fast_train_T2.0_a0.75_speed15_verified"
TEMPERATURE = 2.0                 # soft-target distribution smoothing
DISTILL_ALPHA = 0.75              # emphasize target-distribution alignment
SHOW_FIGURES = True

# Smaller but still non-trivial Transformer.  This is the key runtime knob.
TARGET_MODEL = dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024)
DRAFT_MODEL = dict(d_model=80, n_layers=2, n_heads=4, d_ff=240)

TRAIN_CONFIG = TrainConfig(
    # prompt (~18) + NEW_TOKENS (64) < seq_len (96), so decoding remains
    # inside the context-length distribution seen during training.
    seed=20260820, seq_len=96, batch_size=12,
    target_steps=220, draft_sft_steps=120, distill_steps=220,
    learning_rate=0.0025, min_learning_rate=0.00025,
    weight_decay=0.01, temperature=TEMPERATURE,
    two_stage_distillation=False,
    ce_refinement_fraction=0.0,
    distill_alpha=DISTILL_ALPHA, eval_batches=6, torch_threads=4,
)
CANDIDATE_LENGTHS = (4, 8, 12, 16, 20, 24, 28, 32)
NEW_TOKENS = 64
BENCHMARK_REPEATS = 2
# ============================================================================


def _target_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, max_seq_len=128, **TARGET_MODEL)


def _draft_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(vocab_size=vocab_size, max_seq_len=128, **DRAFT_MODEL)


def main() -> None:
    training.target_config = _target_config
    training.draft_config = _draft_config
    root = Path(__file__).absolute().parent
    out = root / "experiments" / EXPERIMENT_NAME
    # Do not overwrite artifacts or PNG files that may still be open in
    # Windows Photo Viewer.  A new run receives a deterministic suffix.
    if out.exists():
        for index in range(2, 1000):
            candidate = root / "experiments" / f"{EXPERIMENT_NAME}_run{index}"
            if not candidate.exists():
                out = candidate
                break
    artifacts, figures = out / "artifacts", out / "figures"
    started = time.perf_counter()
    print("=" * 72)
    print(f"Fast CPU training: T={TEMPERATURE}, alpha={DISTILL_ALPHA}")
    print(f"Target={TARGET_MODEL}; Draft={DRAFT_MODEL}")
    print("=" * 72)
    train_project(artifacts, TRAIN_CONFIG)
    correctness = run_correctness_checks(artifacts)
    save_json(artifacts / "correctness.json", correctness)
    benchmark = run_benchmark(artifacts, NEW_TOKENS, BENCHMARK_REPEATS, 1,
                              CANDIDATE_LENGTHS, 1)
    save_json(artifacts / "benchmark.json", benchmark)
    print(format_benchmark(benchmark))
    paths = generate_plots(artifacts, figures, artifacts / "benchmark.json")
    print(format_summary(build_summary(artifacts)))
    elapsed = time.perf_counter() - started
    print(f"total wall time: {elapsed:.1f}s (target < 180s)")
    if SHOW_FIGURES:
        # Route through the shared viewer helper so SPECFORGE_NO_PLOT_POPUP=1
        # is honoured here exactly as it is in run_all.py and sweep.py.
        print(f"display mode: {show_plot_files(paths, block=False)}")


if __name__ == "__main__":
    main()
