from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utils import load_json


def build_summary(artifact_dir: str | Path) -> dict:
    artifact_dir = Path(artifact_dir)
    manifest = load_json(artifact_dir / "manifest.json")
    benchmark = load_json(artifact_dir / "benchmark.json")
    correctness = load_json(artifact_dir / "correctness.json")
    best = max(benchmark["speculative"], key=lambda row: row["wall_clock_speedup"])
    metrics = manifest["metrics"]
    return {
        "model": {
            "target_parameters": manifest["target_parameters"],
            "draft_parameters": manifest["draft_parameters"],
            "compression_ratio": manifest["parameter_compression_ratio"],
        },
        "training": {
            "target_validation_loss": [metrics["target_validation_loss_before"], metrics["target_validation_loss_after"]],
            "draft_validation_loss": [metrics["draft_before_sft"]["draft_validation_loss"], metrics["draft_after_distillation"]["draft_validation_loss"]],
            "token_top1_agreement_after_sft": metrics["draft_after_sft"]["token_top1_agreement"],
            "token_top1_agreement_after_distillation": metrics["draft_after_distillation"]["token_top1_agreement"],
            "kl_after_sft": metrics["draft_after_sft"]["kl_target_to_draft"],
            "kl_after_distillation": metrics["draft_after_distillation"]["kl_target_to_draft"],
            "wall_seconds": metrics["training_wall_seconds"],
        },
        "best_benchmark": best,
        "best_request_telemetry": best.get("request_telemetry", {}),
        "baseline_tokens_per_second": benchmark["baseline"]["tokens_per_second"],
        "correctness": correctness,
        "benchmark_environment": benchmark["environment"],
    }


def format_summary(summary: dict) -> str:
    model, training, best = summary["model"], summary["training"], summary["best_benchmark"]
    lines = [
        f"target parameters: {model['target_parameters']:,}",
        f"draft parameters:  {model['draft_parameters']:,}",
        f"compression ratio: {model['compression_ratio']:.2f}x",
        f"target val loss:   {training['target_validation_loss'][0]:.4f} -> {training['target_validation_loss'][1]:.4f}",
        f"draft val loss:    {training['draft_validation_loss'][0]:.4f} -> {training['draft_validation_loss'][1]:.4f}",
        f"agreement:         {training['token_top1_agreement_after_sft']:.2%} -> {training['token_top1_agreement_after_distillation']:.2%}",
        f"KL(T||S):          {training['kl_after_sft']:.4f} -> {training['kl_after_distillation']:.4f}",
        f"best K:            {best['draft_k']}",
        f"acceptance:        {best['mean_acceptance_rate']:.2%}",
        f"target call cut:   {best['target_call_reduction']:.2%}",
        f"throughput:        {summary['baseline_tokens_per_second']:.2f} -> {best['tokens_per_second']:.2f} token/s",
        f"wall speedup:      {best['wall_clock_speedup']:.3f}x",
        f"greedy exact:      {best['greedy_exact_match_rate']:.2%}",
    ]
    telemetry = summary.get("best_request_telemetry")
    if telemetry:
        latency = telemetry["latency_seconds"]
        ttft = telemetry["time_to_first_token_seconds"]
        health = telemetry["decoder_health"]
        cache = telemetry["cache_peak_tokens"]
        lines.extend([
            "request telemetry:",
            f"  latency P50/P95: {latency['p50']:.4f}/{latency['p95']:.4f}s",
            f"  TTFT P50/P95:    {ttft['p50']:.4f}/{ttft['p95']:.4f}s",
            f"  rollback/round:  {health['rollback_rate_per_round']:.2%}",
            f"  cache peak:      target={cache['target_max']}, draft={cache['draft_max']} tokens",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics directly from saved artifacts")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = build_summary(args.artifact_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else format_summary(summary))


if __name__ == "__main__":
    main()
