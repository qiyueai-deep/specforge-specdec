from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import TokenCorpus
from .decoding import autoregressive_generate, speculative_generate
from .training import load_artifacts


@torch.inference_mode()
def run_correctness_checks(artifact_dir: str | Path) -> dict:
    target, draft, tokenizer, manifest = load_artifacts(artifact_dir)
    torch.set_num_threads(int(manifest["train_config"]["torch_threads"]))
    corpus = TokenCorpus.create(int(manifest["train_config"]["seed"]))
    prompts = corpus.validation_prompts(count=10, prompt_len=16)
    greedy_cases = []
    for draft_k in (1, 2, 4, 6):
        matches = 0
        for prompt in prompts:
            baseline, _ = autoregressive_generate(target, prompt, 28, mode="greedy")
            speculative, stats = speculative_generate(target, draft, prompt, 28, draft_k=draft_k, mode="greedy")
            matches += int(baseline == speculative)
        greedy_cases.append({"draft_k": draft_k, "exact_matches": matches, "total": len(prompts)})

    # One-step identity behind rejection sampling:
    # min(p,q) + (1-sum(min(p,q))) * normalize(max(p-q,0)) == p.
    generator = torch.Generator().manual_seed(991)
    max_error = 0.0
    for _ in range(100):
        p = torch.softmax(torch.randn(23, generator=generator), dim=-1)
        q = torch.softmax(torch.randn(23, generator=generator), dim=-1)
        accepted_mass = torch.minimum(p, q)
        reject_mass = 1.0 - accepted_mass.sum()
        residual = torch.clamp(p - q, min=0.0)
        residual = residual / residual.sum().clamp_min(1e-12)
        recovered = accepted_mass + reject_mass * residual
        max_error = max(max_error, float((recovered - p).abs().max()))

    result = {
        "greedy_exact_equivalence": greedy_cases,
        "all_greedy_cases_pass": all(row["exact_matches"] == row["total"] for row in greedy_cases),
        "rejection_sampling_identity_max_abs_error": max_error,
        "rejection_sampling_identity_pass": max_error < 1e-6,
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exact greedy output and rejection-sampling identity")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output", default="artifacts/correctness.json")
    args = parser.parse_args()
    result = run_correctness_checks(args.artifact_dir)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_greedy_cases_pass"] or not result["rejection_sampling_identity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
