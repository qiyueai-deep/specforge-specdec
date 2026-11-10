from __future__ import annotations

import argparse
import json

import torch

from .decoding import autoregressive_generate, speculative_generate
from .training import load_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with autoregressive or speculative decoding")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--prompt", default="推测解码通过")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compare", action="store_true", help="Also run target-only autoregressive decoding")
    args = parser.parse_args()
    target, draft, tokenizer, manifest = load_artifacts(args.artifact_dir)
    torch.set_num_threads(int(manifest["train_config"]["torch_threads"]))
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    tokens, stats = speculative_generate(
        target, draft, prompt_ids, args.max_new_tokens, args.draft_k,
        args.mode, args.temperature, args.top_k, args.seed,
    )
    result = {
        "decoder": "speculative", "mode": args.mode, "draft_k": args.draft_k,
        "text": tokenizer.decode(tokens), "stats": stats.to_dict(),
    }
    if args.compare:
        baseline, baseline_stats = autoregressive_generate(
            target, prompt_ids, args.max_new_tokens, args.mode,
            args.temperature, args.top_k, args.seed,
        )
        result["baseline_text"] = tokenizer.decode(baseline)
        result["greedy_exact_match"] = args.mode != "greedy" or baseline == tokens
        result["baseline_stats"] = baseline_stats.to_dict()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
