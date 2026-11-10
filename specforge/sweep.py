from __future__ import annotations

import argparse
from .benchmark import format_benchmark, run_benchmark
from .plots import generate_plots
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Cheap decoding-parameter sweep without retraining models")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--candidate-lengths", default="1,2,4,8,12,16")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; 0 = auto")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: 3 prompts, 2 repeats, 1 warmup (K range unchanged)")
    parser.add_argument("--output", default="artifacts/benchmark_tuned.json")
    parser.add_argument("--figure-dir", default="figures_tuned")
    args = parser.parse_args()
    values = tuple(int(item.strip()) for item in args.candidate_lengths.split(",") if item.strip())
    repeats = args.repeats
    prompt_count = args.prompt_count
    if args.quick:
        # Same contract as benchmark.py --quick: shrink prompts/repeats/warmups
        # only.  Overriding candidate_lengths here would make the best K
        # incomparable with the official schedule recorded in the README.
        prompt_count = 3
        repeats = 2
    result = run_benchmark(
        args.artifact_dir, max_new_tokens=args.new_tokens, repeats=repeats,
        warmups=1 if args.quick else 2,
        candidate_lengths=values, benchmark_threads=args.threads, prompt_count=prompt_count,
    )
    save_json(args.output, result)
    print(format_benchmark(result))
    generate_plots(args.artifact_dir, args.figure_dir, args.output)
    print(f"tuned figures: {args.figure_dir}")


if __name__ == "__main__":
    main()
