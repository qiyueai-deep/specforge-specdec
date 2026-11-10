from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path

from specforge.benchmark import format_benchmark, run_benchmark
from specforge.config import TrainConfig
from specforge.correctness import run_correctness_checks
from specforge.decoding import speculative_generate
from specforge.training import load_artifacts, train_project
from specforge.utils import save_json
from specforge.plots import generate_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete SpecForge validation pipeline")
    parser.add_argument("--retrain", action="store_true", help="Retrain target and draft even if artifacts exist")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: tiny training schedule + small benchmark")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).absolute().parent
    artifacts = root / "artifacts"
    os.environ.setdefault("SPECFORGE_NO_PLOT_POPUP", "1")
    if args.retrain or not (artifacts / "target.pt").exists():
        print("[1/6] 训练目标模型、SFT 草稿模型并进行知识蒸馏")
        if args.quick:
            cfg = TrainConfig()
            cfg.target_steps = 20
            cfg.draft_sft_steps = 10
            cfg.distill_steps = 20
            cfg.eval_batches = 2
            train_project(artifacts, cfg)
        else:
            train_project(artifacts)
    else:
        print("[1/6] 使用压缩包内已训练模型（传入 --retrain 可从零训练）")

    print("[2/6] 验证贪婪输出严格等价与 rejection sampling 恒等式")
    correctness = run_correctness_checks(artifacts)
    save_json(artifacts / "correctness.json", correctness)
    if not correctness["all_greedy_cases_pass"] or not correctness["rejection_sampling_identity_pass"]:
        raise SystemExit(1)

    print("[3/6] 运行 CPU 性能基准")
    bench_repeats = 2 if args.quick else args.benchmark_repeats
    bench_prompts = 3 if args.quick else 6
    benchmark = run_benchmark(
        artifacts, repeats=bench_repeats, benchmark_threads=1, prompt_count=bench_prompts,
    )
    benchmark_path = artifacts / ("benchmark.json" if args.retrain else "benchmark_latest.json")
    save_json(benchmark_path, benchmark)
    print(format_benchmark(benchmark))

    print("[4/6] 运行单元测试")
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

    print("[5/6] 生成训练与性能曲线")
    # Plot and summarise from the *same* benchmark file we just wrote, so a
    # cached run never silently redraws an older benchmark.json.
    for path in generate_plots(artifacts, root / "figures", benchmark_path):
        print(f"      {path}")

    print("[6/6] 展示一次推测解码")
    target, draft, tokenizer, _ = load_artifacts(artifacts)
    prompt = tokenizer.encode("推测解码通过", add_bos=True)
    tokens, stats = speculative_generate(target, draft, prompt, 36, draft_k=4, mode="greedy")
    print(json.dumps({"text": tokenizer.decode(tokens), "stats": stats.to_dict()}, ensure_ascii=False, indent=2))
    print("\n全部检查通过。")


if __name__ == "__main__":
    main()
