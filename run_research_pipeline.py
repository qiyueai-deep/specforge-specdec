"""One-command reproduction of the v3 research pipeline.

Every step writes its own report into analysis/. Steps are independent and can
be run selectively with --only. The official benchmark is split into disjoint
calibration and held-out prompt sets before K selection.

    python run_research_pipeline.py
    python run_research_pipeline.py --only theory adaptive equivalence
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parent
PY = sys.executable
ART = "experiments/fast_train_T2.0_a0.75_speed15_verified/artifacts"
KS = "4,8,12,16,20,24,28,32"

STEPS: list[tuple[str, list[str]]] = [
    ("theory", [str(ROOT / "analyze_acceptance_theory.py")]),
    ("adaptive", [str(ROOT / "adaptive_k_benchmark.py")]),
    ("distill", [str(ROOT / "run_distill_ablation.py")]),
    ("metrics", [str(ROOT / "tv_kl_report.py")]),
    ("equivalence", [str(ROOT / "run_equivalence_test.py")]),
    ("entropy", [str(ROOT / "entropy_stress_test.py")]),
    ("attribution", [str(ROOT / "run_attribution_control.py")]),
    ("calibration", ["-m", "specforge.benchmark", "--artifact-dir", ART,
                     "--new-tokens", "64", "--repeats", "5", "--prompt-count", "6",
                     "--candidate-lengths", KS, "--split", "calibration",
                     "--output", "artifacts/benchmark_calibration.json"]),
    ("heldout", ["-m", "specforge.benchmark", "--artifact-dir", ART,
                 "--new-tokens", "64", "--repeats", "5", "--prompt-count", "6",
                 "--candidate-lengths", KS, "--split", "held_out",
                 "--output", "artifacts/benchmark_held_out.json"]),
    ("policy", ["-m", "specforge.calibrate_k",
                "--calibration", "artifacts/benchmark_calibration.json",
                "--held-out", "artifacts/benchmark_held_out.json",
                "--output", "artifacts/k_policy_calibration.json"]),
    # Stochastic mode: greedy is the correctness anchor (exact sequence
    # equivalence), sample mode is the second official benchmark whose
    # correctness criterion is the distributional equivalence test above.
    ("calibration_sample", ["-m", "specforge.benchmark", "--artifact-dir", ART,
                            "--new-tokens", "64", "--repeats", "5", "--prompt-count", "6",
                            "--candidate-lengths", KS, "--split", "calibration",
                            "--mode", "sample", "--temperature", "1.0",
                            "--output", "artifacts/benchmark_calibration_sample.json"]),
    ("heldout_sample", ["-m", "specforge.benchmark", "--artifact-dir", ART,
                        "--new-tokens", "64", "--repeats", "5", "--prompt-count", "6",
                        "--candidate-lengths", KS, "--split", "held_out",
                        "--mode", "sample", "--temperature", "1.0",
                        "--output", "artifacts/benchmark_held_out_sample.json"]),
    ("policy_sample", ["-m", "specforge.calibrate_k",
                       "--calibration", "artifacts/benchmark_calibration_sample.json",
                       "--held-out", "artifacts/benchmark_held_out_sample.json",
                       "--output", "artifacts/k_policy_sample.json"]),
    ("tests", ["-m", "unittest", "discover", "-s", "tests", "-q"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SpecForge v3 research pipeline")
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of: " + ", ".join(name for name, _ in STEPS))
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    selected = set(args.only) if args.only else None

    (ROOT / "analysis").mkdir(exist_ok=True)
    failures = []
    for name, command in STEPS:
        if selected is not None and name not in selected:
            continue
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        result = subprocess.run([PY, *command], cwd=str(ROOT), check=False)
        elapsed = time.perf_counter() - started
        print(f"=== {name}: exit={result.returncode}, {elapsed:.1f}s ===", flush=True)
        if result.returncode != 0:
            failures.append(name)
            if args.stop_on_error:
                break
    print("\n=== summary ===")
    if failures:
        print("FAILED:", ", ".join(failures))
        raise SystemExit(1)
    print("all selected steps completed")


if __name__ == "__main__":
    main()
