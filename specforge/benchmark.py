"""Offline benchmark with calibration / held-out separation and honest accounting.

v3 changes:
* ``prompt_split`` selects disjoint calibration and held-out prompt sets, so
  K selection never sees the final test prompts.
* Acceptance rate is the risk-set estimator (see specforge.stats); the legacy
  accepted/proposed number is retained under a separate audit field.
* ``target_call_reduction`` is kept for compatibility but a token-weighted
  ``target_token_ratio`` is added, because one 28-token verification forward
  and one 1-token bonus forward are not the same cost.
* Speedups carry a paired bootstrap confidence interval over prompts.
* ``mode`` can be "sample", so stochastic decoding has actual performance data.
* Execution order rotates within each repeat to remove CPU-frequency drift.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from pathlib import Path

import torch

from .data import TokenCorpus
from .decoding import autoregressive_generate, speculative_generate
from .observability import aggregate_request_telemetry, format_observability
from .training import load_artifacts
from .utils import save_json


def select_prompts(corpus, prompt_count: int, split: str) -> list[list[int]]:
    if split == "all":
        return corpus.validation_prompts(count=prompt_count, prompt_len=18)
    doubled = corpus.validation_prompts(count=prompt_count * 2, prompt_len=18)
    if split == "calibration":
        return doubled[:prompt_count]
    if split == "held_out":
        return doubled[prompt_count:]
    raise ValueError("split must be all / calibration / held_out")


def _speedup_ci(baseline: list[float], policy: list[float], n_boot: int = 1000, seed: int = 0):
    rng = random.Random(seed)
    ratios = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(baseline)) for _ in baseline]
        num = sum(baseline[i] for i in idx)
        den = sum(policy[i] for i in idx)
        ratios.append(num / max(1e-9, den))
    ratios.sort()
    return [ratios[int(0.025 * len(ratios))],
            ratios[min(len(ratios) - 1, int(0.975 * len(ratios)))] ]


def _measure_once(callable_fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = callable_fn()
    return time.perf_counter() - start, result


def run_benchmark(
    artifact_dir: str | Path,
    max_new_tokens: int = 32,
    repeats: int = 5,
    warmups: int = 2,
    candidate_lengths: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
    benchmark_threads: int = 1,
    prompt_count: int = 6,
    use_compile: bool = False,
    prompt_split: str = "all",
    mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
) -> dict:
    target, draft, tokenizer, manifest = load_artifacts(artifact_dir)
    threads = benchmark_threads or min(8, max(1, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    corpus = TokenCorpus.create(int(manifest["train_config"]["seed"]))
    prompts = select_prompts(corpus, prompt_count, prompt_split)

    labels = ["baseline"] + [f"k={value}" for value in candidate_lengths]
    durations_by_label: dict[str, list[float]] = {label: [] for label in labels}
    results_by_label: dict[str, list[object]] = {label: [] for label in labels}
    telemetry_durations: dict[str, list[float]] = {label: [] for label in labels}
    telemetry_stats: dict[str, list[object]] = {label: [] for label in labels}

    def decode_kwargs(k: int | None) -> dict:
        common = dict(mode=mode, temperature=temperature, top_k=top_k,
                      use_compile=use_compile)
        if k is None:
            return common
        return {**common, "draft_k": k}

    for prompt in prompts:
        calls = {
            "baseline": lambda p=prompt: autoregressive_generate(
                target, p, max_new_tokens, **decode_kwargs(None)
            ),
            **{
                f"k={value}": (
                    lambda p=prompt, k=value: speculative_generate(
                        target, draft, p, max_new_tokens, **decode_kwargs(k)
                    )
                )
                for value in candidate_lengths
            },
        }
        for _ in range(warmups):
            for label in labels:
                calls[label]()
        prompt_durations: dict[str, list[float]] = {label: [] for label in labels}
        prompt_results: dict[str, object] = {}
        for repeat in range(repeats):
            order = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
            for label in order:
                seconds, value = _measure_once(calls[label])
                prompt_durations[label].append(seconds)
                prompt_results[label] = value
                telemetry_durations[label].append(seconds)
                telemetry_stats[label].append(value[1])
        for label in labels:
            durations_by_label[label].append(statistics.median(prompt_durations[label]))
            results_by_label[label].append(prompt_results[label])

    baseline_total = sum(durations_by_label["baseline"])
    baseline_runs = results_by_label["baseline"]
    baseline_samples = durations_by_label["baseline"]
    rows = []
    for candidate_length in candidate_lengths:
        label = f"k={candidate_length}"
        results = results_by_label[label]
        total = sum(durations_by_label[label])
        acceptances = [value[1].acceptance_rate for value in results]
        naive = [value[1].naive_acceptance_rate for value in results]
        censored = [value[1].censored_fraction for value in results]
        target_calls = [value[1].target_decode_calls for value in results]
        target_tokens = [value[1].target_processed_tokens for value in results]
        exact_matches = sum(int(value[0] == baseline[0]) for value, baseline in zip(results, baseline_runs))
        rollback_events = sum(value[1].rollback_events for value in results)
        rows.append({
            "draft_k": candidate_length,
            "median_total_seconds": total,
            "tokens_per_second": len(prompts) * max_new_tokens / total,
            "wall_clock_speedup": baseline_total / total,
            "speedup_ci95": _speedup_ci(baseline_samples, durations_by_label[label], seed=candidate_length),
            "mean_acceptance_rate": sum(acceptances) / len(acceptances),
            "mean_naive_acceptance_rate": sum(naive) / len(naive),
            "mean_censored_fraction": sum(censored) / len(censored),
            "mean_target_decode_calls": sum(target_calls) / len(target_calls),
            "target_call_reduction": 1.0 - (sum(target_calls) / len(target_calls)) / max_new_tokens,
            # Token-weighted cost: a K-token verification forward and a 1-token
            # bonus forward are both counted once above, which is misleading.
            "target_token_ratio": (sum(target_tokens) / len(target_tokens)) / max_new_tokens,
            "greedy_exact_match_rate": exact_matches / len(prompts),
            "rollback_events": rollback_events,
            "request_telemetry": aggregate_request_telemetry(telemetry_stats[label], telemetry_durations[label]),
        })
    result = {
        "environment": {
            "torch_version": torch.__version__, "torch_threads": threads,
            "device": "cpu", "prompt_count": len(prompts), "prompt_tokens": 18,
            "prompt_split": prompt_split, "new_tokens": max_new_tokens,
            "repeats": repeats, "warmups": warmups, "mode": mode,
            "temperature": temperature, "top_k": top_k, "use_compile": use_compile,
            "acceptance_definition": "accepted / verified (risk set)",
        },
        "baseline": {
            "median_total_seconds": baseline_total,
            "tokens_per_second": len(prompts) * max_new_tokens / baseline_total,
            "mean_target_decode_calls": max_new_tokens,
            "request_telemetry": aggregate_request_telemetry(
                telemetry_stats["baseline"], telemetry_durations["baseline"]
            ),
        },
        "speculative": rows,
    }
    return result


def format_benchmark(result: dict) -> str:
    baseline = result["baseline"]
    lines = [
        f"autoregressive: {baseline['tokens_per_second']:.2f} token/s, "
        f"target calls={baseline['mean_target_decode_calls']:.1f}",
        "K | accept | naive | censored | calls | token ratio | token/s | speedup | 95% CI | exact",
    ]
    for row in result["speculative"]:
        ci = row["speedup_ci95"]
        lines.append(
            f"{row['draft_k']:>1} | {row['mean_acceptance_rate']:>6.2%} | "
            f"{row['mean_naive_acceptance_rate']:>6.2%} | "
            f"{row['mean_censored_fraction']:>8.1%} | "
            f"{row['mean_target_decode_calls']:>5.1f} | "
            f"{row['target_token_ratio']:>11.2%} | "
            f"{row['tokens_per_second']:>7.2f} | {row['wall_clock_speedup']:>7.3f}x | "
            f"[{ci[0]:.2f},{ci[1]:.2f}] | {row['greedy_exact_match_rate']:>6.2%}"
        )
    telemetry = {"baseline": baseline["request_telemetry"]}
    telemetry.update({f"k={row['draft_k']}": row["request_telemetry"] for row in result["speculative"]})
    return "\n".join(lines) + "\n\n" + format_observability({"request_telemetry": telemetry})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark autoregressive and speculative decoding")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; 0 = auto (min(8, cpu_count))")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--candidate-lengths", default="1,2,4,8,12,16")
    parser.add_argument("--split", choices=("all", "calibration", "held_out"), default="all",
                        help="Disjoint prompt sets; K must be chosen on calibration only")
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--use-compile", action="store_true",
                        help="Use the torch.compile functional-forward decode path (needs a working CPU compiler)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: 3 prompts, 2 repeats, warmups=1")
    args = parser.parse_args()
    candidate_lengths = tuple(int(value.strip()) for value in args.candidate_lengths.split(",") if value.strip())
    repeats = args.repeats
    prompt_count = args.prompt_count
    warmups = 2
    if args.quick:
        prompt_count = 3
        repeats = 2
        warmups = 1
    result = run_benchmark(
        args.artifact_dir, max_new_tokens=args.new_tokens, repeats=repeats, warmups=warmups,
        candidate_lengths=candidate_lengths, benchmark_threads=args.threads, prompt_count=prompt_count,
        use_compile=args.use_compile, prompt_split=args.split, mode=args.mode,
        temperature=args.temperature, top_k=args.top_k,
    )
    save_json(args.output, result)
    print(format_benchmark(result))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
