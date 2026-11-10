"""PyCharm one-click Hugging Face real-model assisted-generation benchmark.

First run downloads model weights. This optional experiment is independent from
the under-three-minute tiny-model training entry in pycharm_fast_train.py.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import time

from specforge.hf_benchmark import (
    HFBenchmarkConfig,
    format_hf_benchmark,
    run_hf_benchmark,
    save_hf_report,
)
from specforge.hf_models import HFModelConfig, download_hf_models
from specforge.plots import show_plot_files


# =============================== CONFIGURATION ==============================
MODEL_CONFIG = HFModelConfig(
    target_name="Qwen/Qwen3-1.7B",
    draft_name="Qwen/Qwen3-0.6B",
    device="auto",              # auto -> CUDA when available, otherwise CPU
    local_files_only=False,      # set True after the models are cached locally
    download_provider="modelscope",  # avoids Hugging Face timeout in this network
)
BENCHMARK_CONFIG = HFBenchmarkConfig(
    # CPU-safe default: normally finishes well below three minutes after the
    # one-time model download.  Increase these values only for a formal run.
    candidate_lengths=(4, 8, 10, 12, 14, 16, 20, 24),
    max_new_tokens=8,
    repeats=1,
    warmups=0,
    prompt_count=1,
    enable_thinking=False,       # required for deterministic greedy equivalence
)
SHOW_FIGURE = True
PREDOWNLOAD_MODELS = True       # reliable serial HTTPS + automatic resume
# ============================================================================


def main() -> None:
    root = Path(__file__).absolute().parent
    output = root / "experiments" / "huggingface_qwen3_real_models"
    if output.exists():
        for index in range(2, 1000):
            candidate = root / "experiments" / f"huggingface_qwen3_real_models_run{index}"
            if not candidate.exists():
                output = candidate
                break
    print("=" * 72)
    print("SpecForge Hugging Face real-model benchmark")
    print(f"Target: {MODEL_CONFIG.target_name}")
    print(f"Draft:  {MODEL_CONFIG.draft_name}")
    print("First run may spend substantial time downloading model weights.")
    print("Cached CPU quick benchmark is configured to finish in about 1-2 minutes.")
    print("=" * 72)
    started = time.perf_counter()
    runtime_config = MODEL_CONFIG
    if PREDOWNLOAD_MODELS:
        target_path, draft_path = download_hf_models(MODEL_CONFIG)
        runtime_config = replace(
            MODEL_CONFIG,
            target_name=target_path,
            draft_name=draft_path,
            local_files_only=True,
        )
    result = run_hf_benchmark(runtime_config, BENCHMARK_CONFIG)
    result["source_model_config"] = MODEL_CONFIG.to_dict()
    json_path, report_path, figure_path = save_hf_report(result, output)
    print(format_hf_benchmark(result))
    print(f"total wall time: {time.perf_counter() - started:.1f}s")
    print(f"JSON:   {json_path}")
    print(f"Report: {report_path}")
    print("Figures:")
    print(f"  {figure_path}")
    print(f"  {output / 'hf_latency_comparison.png'}")
    print(f"  {output / 'hf_model_validation.png'}")
    if SHOW_FIGURE:
        # Same viewer helper as the other entry points, so SPECFORGE_NO_PLOT_POPUP=1
        # suppresses these windows too instead of being silently ignored.
        figures = [figure_path, output / "hf_latency_comparison.png", output / "hf_model_validation.png"]
        print(f"display mode: {show_plot_files(figures, block=False)}")


if __name__ == "__main__":
    main()
