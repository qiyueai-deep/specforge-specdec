"""Hardware/model aware K selection with a real calibration / held-out split.

v1 was a stub: it read the *main* benchmark.json (the same file used to report
the headline speedup) and took argmax over its speedups. That is exactly
selecting K on the test set, while PROJECT_BRIEF.md claimed the opposite, and
no k_policy.json was ever produced.

v2 refuses to proceed unless the calibration file was produced with
``--split calibration`` and the held-out file with ``--split held_out``. It
reports the held-out speedup of the *calibration-selected* K, plus the regret
against a (diagnostic-only) held-out oracle, so overfitting is measurable.

Usage:
    python -m specforge.benchmark --split calibration --output artifacts/benchmark_calibration.json
    python -m specforge.benchmark --split held_out  --output artifacts/benchmark_held_out.json
    python -m specforge.calibrate_k --calibration artifacts/benchmark_calibration.json \
        --held-out artifacts/benchmark_held_out.json --output artifacts/k_policy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _split_of(data: dict) -> str:
    return data.get("environment", {}).get("prompt_split", "unknown")


def build_policy(calibration: dict, held_out: dict | None, calibration_name: str = "") -> dict:
    cal_split = _split_of(calibration)
    if cal_split != "calibration":
        raise ValueError(
            f"calibration file has prompt_split={cal_split!r}; "
            "run benchmark.py with --split calibration"
        )
    rows = calibration["speculative"]
    if not rows:
        raise ValueError("calibration file has no speculative rows")
    best = max(rows, key=lambda x: x["wall_clock_speedup"])
    selected_k = best["draft_k"]

    policy = {
        "selection_rule": "argmax wall_clock_speedup on calibration prompts only",
        "selected_k": selected_k,
        "calibration_split": cal_split,
        "calibration_file": calibration_name,
        "calibration_speedup": best["wall_clock_speedup"],
        "calibration_speedup_ci95": best.get("speedup_ci95"),
        "calibration_prompt_count": calibration.get("environment", {}).get("prompt_count"),
        "candidates": [
            {"k": x["draft_k"], "speedup": x["wall_clock_speedup"],
             "acceptance": x["mean_acceptance_rate"],
             "speedup_ci95": x.get("speedup_ci95")}
            for x in rows
        ],
        "held_out": None,
    }

    if held_out is not None:
        ho_split = _split_of(held_out)
        if ho_split != "held_out":
            raise ValueError(
                f"held-out file has prompt_split={ho_split!r}; "
                "run benchmark.py with --split held_out"
            )
        match = next((r for r in held_out["speculative"] if r["draft_k"] == selected_k), None)
        if match is None:
            raise ValueError(f"selected K={selected_k} is absent from the held-out benchmark")
        oracle = max(held_out["speculative"], key=lambda x: x["wall_clock_speedup"])
        policy["held_out"] = {
            "held_out_split": ho_split,
            "speedup": match["wall_clock_speedup"],
            "speedup_ci95": match.get("speedup_ci95"),
            "acceptance_rate": match["mean_acceptance_rate"],
            "greedy_exact_match_rate": match["greedy_exact_match_rate"],
            "prompt_count": held_out.get("environment", {}).get("prompt_count"),
            # Diagnostic only. The oracle K is computed on held-out data and
            # must never be used as the reported policy.
            "oracle_k_diagnostic_only": oracle["draft_k"],
            "oracle_speedup_diagnostic_only": oracle["wall_clock_speedup"],
            "regret_vs_oracle": oracle["wall_clock_speedup"] - match["wall_clock_speedup"],
        }
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Select K on calibration, report on held-out")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--held-out", default=None)
    parser.add_argument("--output", default="artifacts/k_policy.json")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    held_out = (json.loads(Path(args.held_out).read_text(encoding="utf-8"))
                if args.held_out else None)
    policy = build_policy(calibration, held_out, args.calibration)
    if held_out is None:
        print("WARNING: no held-out file supplied; policy has no held-out validation.")
    Path(args.output).write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
