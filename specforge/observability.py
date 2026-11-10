"""Production-style request telemetry aggregation for offline benchmarks.

The project is CPU-only, so cache occupancy is reported in token positions rather
than pretending to expose GPU VRAM.  This schema can be exported unchanged to a
metrics backend when the decoder is later served behind an API.
"""

from __future__ import annotations

import math
from typing import Iterable

from .decoding import GenerationStats


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile with no third-party dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_request_telemetry(stats: list[GenerationStats], durations: list[float]) -> dict:
    """Aggregate request-level latency, cache and decoder-health metrics."""
    if len(stats) != len(durations):
        raise ValueError("stats and durations must have the same number of requests")
    request_count = len(stats)
    total_proposals = sum(item.proposed_draft_tokens for item in stats)
    total_rounds = sum(item.proposal_rounds for item in stats)
    total_rollbacks = sum(item.rollback_events for item in stats)
    return {
        "request_count": request_count,
        "latency_seconds": {
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "mean": sum(durations) / max(1, request_count),
        },
        "time_to_first_token_seconds": {
            "p50": percentile([item.time_to_first_token_seconds for item in stats], 0.50),
            "p95": percentile([item.time_to_first_token_seconds for item in stats], 0.95),
        },
        "prefill_seconds": {
            "p50": percentile([item.prefill_seconds for item in stats], 0.50),
            "p95": percentile([item.prefill_seconds for item in stats], 0.95),
        },
        "cache_peak_tokens": {
            "target_max": max((item.peak_target_cache_tokens for item in stats), default=0),
            "draft_max": max((item.peak_draft_cache_tokens for item in stats), default=0),
        },
        "decoder_health": {
            "acceptance_rate": sum(item.acceptance_rate for item in stats) / max(1, request_count),
            "naive_acceptance_rate_audit_only": (
                sum(item.naive_acceptance_rate for item in stats) / max(1, request_count)
            ),
            "censored_fraction": sum(item.censored_fraction for item in stats) / max(1, request_count),
            "rollback_rate_per_round": total_rollbacks / max(1, total_rounds),
            "rollback_rate_per_proposed_token": total_rollbacks / max(1, total_proposals),
            "mean_target_decode_calls": sum(item.target_decode_calls for item in stats) / max(1, request_count),
        },
    }


def format_observability(result: dict) -> str:
    """Human-readable report suitable for a local benchmark or service log."""
    lines = ["Request telemetry (P50/P95, seconds):"]
    for label, metrics in result["request_telemetry"].items():
        latency, ttft = metrics["latency_seconds"], metrics["time_to_first_token_seconds"]
        health, cache = metrics["decoder_health"], metrics["cache_peak_tokens"]
        lines.append(
            f"{label}: latency={latency['p50']:.4f}/{latency['p95']:.4f}, "
            f"TTFT={ttft['p50']:.4f}/{ttft['p95']:.4f}, "
            f"rollback/round={health['rollback_rate_per_round']:.2%}, "
            f"cache peak={cache['target_max']}/{cache['draft_max']} tokens"
        )
    return "\n".join(lines)
