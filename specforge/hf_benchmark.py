"""Real-model assisted-generation benchmark with request-level telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import logging
import statistics
import threading
import time

import torch

from .hf_models import HFModelConfig, load_hf_models, parameter_count
from .observability import percentile
from .utils import save_json


DEFAULT_PROMPTS = (
    "请简要解释推测解码为什么能够加速大语言模型推理。",
    "知识蒸馏中的温度参数有什么作用？",
    "为什么 KV Cache 在拒绝候选后必须回滚？",
    "请比较吞吐量、首 token 延迟和端到端延迟的区别。",
)


class _AssistedGenerationDeprecationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            "Passing `generation_config` together with generation-related arguments" in message
            and "deprecated" in message
        )


def _suppress_transformers_internal_notice() -> None:
    notice_filter = _AssistedGenerationDeprecationFilter()
    for logger_name in (
        "transformers", "transformers.generation", "transformers.generation.utils",
        "transformers.generation.candidate_generator",
    ):
        logging.getLogger(logger_name).addFilter(notice_filter)
    for handler in logging.getLogger().handlers:
        handler.addFilter(notice_filter)


@dataclass(frozen=True)
class HFBenchmarkConfig:
    candidate_lengths: tuple[int, ...] = (4, 8, 12, 16)
    max_new_tokens: int = 48
    repeats: int = 2
    warmups: int = 1
    prompt_count: int = 4
    seed: int = 20260827
    enable_thinking: bool = False

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.repeats < 1 or self.prompt_count < 1:
            raise ValueError("max_new_tokens, repeats and prompt_count must be positive")
        if not self.candidate_lengths or any(value < 1 for value in self.candidate_lengths):
            raise ValueError("candidate_lengths must contain positive integers")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_lengths"] = list(self.candidate_lengths)
        return payload


class _FirstTokenStreamer:
    """Minimal streamer measuring when generate publishes its first new token."""

    def __init__(self) -> None:
        self.started = 0.0
        self.first_token_seconds: float | None = None
        self._skip_prompt = True

    def put(self, value) -> None:
        if self._skip_prompt:
            self._skip_prompt = False
            return
        if self.first_token_seconds is None:
            self.first_token_seconds = time.perf_counter() - self.started

    def end(self) -> None:
        return None


def _resolve_ttft(first_token_seconds: float | None, elapsed: float) -> float:
    """Fall back to end-to-end latency only when no first-token event arrived.

    A measured 0.0 is a real measurement and must not be replaced by the
    `or` idiom, which treats every falsy value as missing.
    """
    return elapsed if first_token_seconds is None else first_token_seconds


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_memory_mb(device: str) -> float | None:
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _prepare_inputs(tokenizer, prompt: str, device: str, enable_thinking: bool):
    """Use Qwen3 chat format; fall back for simple/mock tokenizers in tests."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        options = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        try:
            inputs = tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **options
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(messages, **options)
        return inputs.to(device)
    return tokenizer(prompt, return_tensors="pt").to(device)


def _measure_generate(model, generation_kwargs: dict, device: str) -> tuple[torch.Tensor, float, float, float | None]:
    streamer = _FirstTokenStreamer()
    holder: dict[str, object] = {}

    def run() -> None:
        try:
            with torch.inference_mode():
                holder["output"] = model.generate(**generation_kwargs, streamer=streamer)
        except BaseException as error:  # propagate worker failures on the caller thread
            holder["error"] = error

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    streamer.started = time.perf_counter()
    worker = threading.Thread(target=run, daemon=False)
    worker.start()
    worker.join()
    _sync(device)
    elapsed = time.perf_counter() - streamer.started
    if "error" in holder:
        raise holder["error"]  # type: ignore[misc]
    output = holder["output"]
    ttft = _resolve_ttft(streamer.first_token_seconds, elapsed)
    return output, elapsed, ttft, _peak_memory_mb(device)  # type: ignore[return-value]


def _aggregate(label: str, durations: list[float], ttfts: list[float], token_counts: list[int],
               peak_memory: list[float], exact_matches: list[bool] | None = None) -> dict:
    total_tokens = sum(token_counts)
    total_seconds = sum(durations)
    return {
        "label": label,
        "request_count": len(durations),
        "latency_seconds": {
            "mean": statistics.fmean(durations),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
        },
        "time_to_first_token_seconds": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
        },
        "tokens_per_second": total_tokens / max(total_seconds, 1e-12),
        "peak_gpu_memory_mb": max(peak_memory) if peak_memory else None,
        "greedy_exact_match_rate": (
            sum(exact_matches) / len(exact_matches) if exact_matches else None
        ),
        # transformers.generate does not expose assistant accept/reject counters.
        "assistant_acceptance_rate": None,
        "rollback_rate": None,
    }


def run_hf_benchmark(model_config: HFModelConfig, benchmark_config: HFBenchmarkConfig,
                     prompts: tuple[str, ...] = DEFAULT_PROMPTS) -> dict:
    _suppress_transformers_internal_notice()
    benchmark_config.validate()
    torch.manual_seed(benchmark_config.seed)
    target, draft, tokenizer, device = load_hf_models(model_config)
    # Qwen3 model cards contain sampling defaults.  This benchmark is strictly
    # greedy, so clear inactive sampling fields instead of letting Transformers
    # emit an "invalid generation flags" warning on every PyCharm run.
    for model in (target, draft):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    selected_prompts = prompts[:benchmark_config.prompt_count]
    if not selected_prompts:
        raise ValueError("No prompts selected")

    records: dict[str, dict[str, list]] = {}
    labels = ["baseline"] + [f"k={value}" for value in benchmark_config.candidate_lengths]
    for label in labels:
        records[label] = {"durations": [], "ttfts": [], "tokens": [], "memory": [], "outputs": []}

    for prompt in selected_prompts:
        inputs = _prepare_inputs(tokenizer, prompt, device, benchmark_config.enable_thinking)
        input_length = int(inputs["input_ids"].shape[1])
        base_kwargs = {
            **inputs,
            "max_new_tokens": benchmark_config.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        for _ in range(benchmark_config.warmups):
            with torch.inference_mode():
                target.generate(**base_kwargs)
                for candidate_length in benchmark_config.candidate_lengths:
                    draft.generation_config.num_assistant_tokens = candidate_length
                    draft.generation_config.num_assistant_tokens_schedule = "constant"
                    target.generate(**base_kwargs, assistant_model=draft)

        for repeat in range(benchmark_config.repeats):
            output, elapsed, ttft, memory = _measure_generate(target, base_kwargs, device)
            records["baseline"]["durations"].append(elapsed)
            records["baseline"]["ttfts"].append(ttft)
            records["baseline"]["tokens"].append(int(output.shape[1]) - input_length)
            records["baseline"]["outputs"].append(output.cpu())
            if memory is not None:
                records["baseline"]["memory"].append(memory)

            for candidate_length in benchmark_config.candidate_lengths:
                draft.generation_config.num_assistant_tokens = candidate_length
                draft.generation_config.num_assistant_tokens_schedule = "constant"
                assisted_kwargs = {**base_kwargs, "assistant_model": draft}
                output, elapsed, ttft, memory = _measure_generate(target, assisted_kwargs, device)
                label = f"k={candidate_length}"
                records[label]["durations"].append(elapsed)
                records[label]["ttfts"].append(ttft)
                records[label]["tokens"].append(int(output.shape[1]) - input_length)
                records[label]["outputs"].append(output.cpu())
                if memory is not None:
                    records[label]["memory"].append(memory)

    baseline_outputs = records["baseline"]["outputs"]
    baseline = _aggregate(
        "baseline", records["baseline"]["durations"], records["baseline"]["ttfts"],
        records["baseline"]["tokens"], records["baseline"]["memory"],
    )
    rows = []
    for candidate_length in benchmark_config.candidate_lengths:
        label = f"k={candidate_length}"
        exact = [torch.equal(value, baseline_value) for value, baseline_value in zip(
            records[label]["outputs"], baseline_outputs
        )]
        row = _aggregate(
            label, records[label]["durations"], records[label]["ttfts"],
            records[label]["tokens"], records[label]["memory"], exact,
        )
        row["draft_k"] = candidate_length
        row["wall_clock_speedup"] = baseline["latency_seconds"]["mean"] / row["latency_seconds"]["mean"]
        rows.append(row)

    return {
        "schema_version": 1,
        "model_config": model_config.to_dict(),
        "benchmark_config": benchmark_config.to_dict(),
        "environment": {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "thinking_mode": benchmark_config.enable_thinking,
        },
        "models": {
            "target_parameters": parameter_count(target),
            "draft_parameters": parameter_count(draft),
        },
        "baseline": baseline,
        "assisted": rows,
        "limitations": [
            "Hugging Face assisted generation does not expose internal acceptance or rollback counters.",
            "Wall-clock results are valid only for the recorded hardware, prompts and generation settings.",
        ],
    }


def format_hf_benchmark(result: dict) -> str:
    environment, baseline = result["environment"], result["baseline"]
    lines = [
        f"device: {environment['device']} ({environment.get('cuda_name') or 'CPU'})",
        f"target/draft parameters: {result['models']['target_parameters']:,} / "
        f"{result['models']['draft_parameters']:,}",
        f"baseline: {baseline['tokens_per_second']:.2f} token/s, "
        f"P50/P95={baseline['latency_seconds']['p50']:.3f}/{baseline['latency_seconds']['p95']:.3f}s",
        "K | token/s | speedup | TTFT P50/P95 | exact",
    ]
    for row in result["assisted"]:
        ttft = row["time_to_first_token_seconds"]
        lines.append(
            f"{row['draft_k']:>2} | {row['tokens_per_second']:>7.2f} | "
            f"{row['wall_clock_speedup']:>7.3f}x | {ttft['p50']:.3f}/{ttft['p95']:.3f}s | "
            f"{row['greedy_exact_match_rate']:.2%}"
        )
    return "\n".join(lines)


def plot_hf_benchmark(result: dict, output_dir: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = result["assisted"]
    k_values = [row["draft_k"] for row in rows]
    speedups = [row["wall_clock_speedup"] for row in rows]
    throughput = [row["tokens_per_second"] for row in rows]
    p50 = [row["latency_seconds"]["p50"] for row in rows]
    ttft = [row["time_to_first_token_seconds"]["p50"] for row in rows]
    baseline = result["baseline"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(k_values, speedups, marker="o", color="#2563eb")
    axes[0, 0].axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Wall-clock speedup", xlabel="Draft K", ylabel="Speedup (x)")
    axes[0, 1].plot(k_values, throughput, marker="o", color="#059669")
    axes[0, 1].axhline(baseline["tokens_per_second"], color="#64748b", linestyle="--", linewidth=1,
                       label=f"baseline: {baseline['tokens_per_second']:.2f}")
    axes[0, 1].set(title="Generation throughput", xlabel="Draft K", ylabel="Token/s")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(k_values, p50, marker="o", color="#d97706")
    axes[1, 0].axhline(baseline["latency_seconds"]["p50"], color="#64748b", linestyle="--", linewidth=1)
    axes[1, 0].set(title="End-to-end latency P50", xlabel="Draft K", ylabel="Seconds")
    axes[1, 1].plot(k_values, ttft, marker="o", color="#7c3aed")
    axes[1, 1].axhline(baseline["time_to_first_token_seconds"]["p50"], color="#64748b", linestyle="--", linewidth=1)
    axes[1, 1].set(title="Time to first token P50", xlabel="Draft K", ylabel="Seconds")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xticks(k_values)
    fig.suptitle("Hugging Face Assisted Generation Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output / "hf_speculative_benchmark.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hf_latency_comparison(result: dict, output_dir: str | Path) -> Path:
    """Compare baseline and assisted latency distributions from measured data."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = result["assisted"]
    labels = ["baseline"] + [f"K={row['draft_k']}" for row in rows]
    latency_p50 = [result["baseline"]["latency_seconds"]["p50"]] + [
        row["latency_seconds"]["p50"] for row in rows
    ]
    latency_p95 = [result["baseline"]["latency_seconds"]["p95"]] + [
        row["latency_seconds"]["p95"] for row in rows
    ]
    ttft_p50 = [result["baseline"]["time_to_first_token_seconds"]["p50"]] + [
        row["time_to_first_token_seconds"]["p50"] for row in rows
    ]
    ttft_p95 = [result["baseline"]["time_to_first_token_seconds"]["p95"]] + [
        row["time_to_first_token_seconds"]["p95"] for row in rows
    ]
    positions = list(range(len(labels)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, p50, p95, title in (
        (axes[0], latency_p50, latency_p95, "End-to-end latency"),
        (axes[1], ttft_p50, ttft_p95, "Time to first token"),
    ):
        axis.bar([x - width / 2 for x in positions], p50, width, label="P50", color="#2563eb")
        axis.bar([x + width / 2 for x in positions], p95, width, label="P95", color="#f97316")
        axis.set(title=title, xlabel="Generation mode", ylabel="Seconds")
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Qwen3 measured latency comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output / "hf_latency_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hf_model_validation(result: dict, output_dir: str | Path) -> Path:
    """Visualize real model scale and greedy-output correctness by K."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = result["assisted"]
    target_params = result["models"]["target_parameters"] / 1e9
    draft_params = result["models"]["draft_parameters"] / 1e9
    compression = target_params / max(draft_params, 1e-12)
    k_values = [row["draft_k"] for row in rows]
    exact = [100.0 * row["greedy_exact_match_rate"] for row in rows]
    latency_reduction = [
        100.0 * (1.0 - row["latency_seconds"]["mean"] / result["baseline"]["latency_seconds"]["mean"])
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    bars = axes[0].bar(["Target", "Draft"], [target_params, draft_params], color=["#2563eb", "#f97316"])
    axes[0].set(title=f"Model scale ({compression:.2f}x compression)", ylabel="Parameters (billions)")
    for bar, value in zip(bars, (target_params, draft_params)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}B", ha="center", va="bottom")
    axes[1].plot(k_values, exact, marker="o", color="#16a34a")
    axes[1].axhline(100.0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set(title="Greedy output exact match", xlabel="Draft K", ylabel="Percent", ylim=(98.5, 100.3))
    axes[2].bar([str(value) for value in k_values], latency_reduction, color="#7c3aed")
    axes[2].axhline(0.0, color="#64748b", linewidth=1)
    axes[2].set(title="Mean latency reduction", xlabel="Draft K", ylabel="Reduction vs baseline (%)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Qwen3 model scale and correctness validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output / "hf_model_validation.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_hf_report(result: dict, output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    json_path = output / "hf_benchmark.json"
    report_path = output / "hf_benchmark_report.md"
    save_json(json_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path = plot_hf_benchmark(result, output)
    latency_path = plot_hf_latency_comparison(result, output)
    validation_path = plot_hf_model_validation(result, output)
    report_path.write_text(
        "# Hugging Face 真实模型推测解码报告\n\n```text\n"
        + format_hf_benchmark(result)
        + "\n```\n\n## Figures\n\n"
        + f"- `{figure_path.name}`：吞吐、加速比、端到端延迟与 TTFT 随 K 的变化。\n"
        + f"- `{latency_path.name}`：baseline 与各 K 的 P50/P95 延迟对比。\n"
        + f"- `{validation_path.name}`：真实模型规模、贪婪输出一致性与平均延迟降幅。\n\n"
        + "> 接受率和回滚率为 `null`，因为官方 assisted generation API 不公开内部计数。\n",
        encoding="utf-8",
    )
    return json_path, report_path, figure_path
