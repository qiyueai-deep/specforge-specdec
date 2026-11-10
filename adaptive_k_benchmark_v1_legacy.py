"""自适应 K（置信度早停）推测解码基准（任务2）。

依据 analyze_acceptance_theory.py 的发现：逐位置接受危险率随深度衰减
（位置 1 ≈1.0 → 位置 25 ≈0.7），深处提议性价比低。
策略：draft 每步先看自己的置信度（softmax 最大概率），低于阈值 θ 即停止
继续起草，提前提交验证块。贪婪模式下与 target-only 输出逐 token 等价。

对比对象：target-only 基线、固定 K=12/28（官方 speculative_generate）。
计时协议与 specforge.benchmark 相同：单线程、6 prompt、64 token、warmup、
逐 prompt 取 median 后求和。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))

from specforge.data import TokenCorpus  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.training import load_artifacts  # noqa: E402

OUT_DIR = ROOT / "analysis"
ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
NEW_TOKENS = 64
PROMPT_COUNT = 6
PROMPT_LEN = 18
REPEATS = 5
WARMUPS = 1
K_MAX = 28
THRESHOLDS = (0.5, 0.7, 0.9)
FIXED_KS = (12, 28)


@torch.inference_mode()
def adaptive_speculative_generate(target, draft, prompt_ids, max_new_tokens,
                                  k_max: int, conf_threshold: float):
    """贪婪推测解码 + 置信度早停。返回 (output, stats_dict)。

    早停规则：在提议第 i+1 个 token 前（i≥1），若 draft 对下一 token 的
    softmax 最大概率 < conf_threshold，则停止起草、提前验证。
    正确性与贪婪推测解码等价（仅验证块更短）。
    """
    target_cache, draft_cache = target.new_cache(), draft.new_cache()
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    target_next = target(prompt, cache=target_cache)[0, -1]
    draft_next = draft(prompt, cache=draft_cache)[0, -1]
    output = list(prompt_ids)
    generated = 0
    stats = {"rounds": 0, "proposed": 0, "accepted": 0, "target_calls": 0,
             "draft_steps": 0, "rollback_events": 0}
    while generated < max_new_tokens:
        remaining = max_new_tokens - generated
        if remaining == 1:
            token = int(target_next.argmax())
            output.append(token)
            token_tensor = torch.tensor([[token]], dtype=torch.long)
            target_next = target(token_tensor, cache=target_cache)[0, -1]
            draft_next = draft(token_tensor, cache=draft_cache)[0, -1]
            generated += 1
            stats["target_calls"] += 1
            break
        proposal_count = min(k_max, remaining - 1)
        base_len = target_cache.length
        proposals: list[int] = []
        for _ in range(proposal_count):
            if proposals:
                confidence = float(F.softmax(draft_next.float(), dim=-1).max())
                if confidence < conf_threshold:
                    break
            token = int(draft_next.argmax())
            proposals.append(token)
            draft_next = draft(torch.tensor([[token]], dtype=torch.long), cache=draft_cache)[0, -1]
            stats["draft_steps"] += 1
        stats["proposed"] += len(proposals)
        stats["rounds"] += 1
        block_logits = target(torch.tensor([proposals], dtype=torch.long), cache=target_cache)[0]
        stats["target_calls"] += 1
        accepted, replacement = 0, None
        for index, token in enumerate(proposals):
            verification = target_next if index == 0 else block_logits[index - 1]
            if token == int(verification.argmax()):
                accepted += 1
                continue
            replacement = int(verification.argmax())
            break
        stats["accepted"] += accepted
        output.extend(proposals[:accepted])
        generated += accepted
        if replacement is not None:
            stats["rollback_events"] += 1
            target_cache.truncate(base_len + accepted)
            draft_cache.truncate(base_len + accepted)
            output.append(replacement)
            generated += 1
            token_tensor = torch.tensor([[replacement]], dtype=torch.long)
            target_next = target(token_tensor, cache=target_cache)[0, -1]
            draft_next = draft(token_tensor, cache=draft_cache)[0, -1]
            stats["target_calls"] += 1
        else:
            target_next = block_logits[-1]
            if generated < max_new_tokens:
                bonus = int(target_next.argmax())
                output.append(bonus)
                generated += 1
                token_tensor = torch.tensor([[bonus]], dtype=torch.long)
                target_next = target(token_tensor, cache=target_cache)[0, -1]
                draft_next = draft(token_tensor, cache=draft_cache)[0, -1]
                stats["target_calls"] += 1
    assert generated == max_new_tokens
    return output, stats


def measure(callable_fn, prompts, repeats: int, warmups: int):
    durations, last_results = [], None
    for prompt in prompts:
        for _ in range(warmups):
            callable_fn(prompt)
        per_prompt = []
        for _ in range(repeats):
            start = time.perf_counter()
            last_results = callable_fn(prompt)
            per_prompt.append(time.perf_counter() - start)
        durations.append(statistics.median(per_prompt))
    return sum(durations), last_results


def main() -> None:
    torch.set_num_threads(1)
    target, draft, _tok, _man = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820)
    prompts = corpus.validation_prompts(count=PROMPT_COUNT, prompt_len=PROMPT_LEN)

    configs: list[tuple[str, object]] = [("baseline", None)]
    for k in FIXED_KS:
        configs.append((f"fixed_k{k}", k))
    for theta in THRESHOLDS:
        configs.append((f"adaptive_t{theta}", theta))

    results = []
    for label, param in configs:
        if param is None:
            def run(prompt, _p=param):
                return autoregressive_generate(target, prompt, NEW_TOKENS, mode="greedy")
        elif isinstance(param, int):
            def run(prompt, _p=param):
                return speculative_generate(target, draft, prompt, NEW_TOKENS, draft_k=_p, mode="greedy")
        else:
            def run(prompt, _p=param):
                return adaptive_speculative_generate(target, draft, prompt, NEW_TOKENS, K_MAX, _p)
        total, last = measure(run, prompts, REPEATS, WARMUPS)
        if label == "baseline":
            baseline_total = total
            results.append({"label": label, "seconds": total, "speedup": 1.0})
            print(f"{label:>16}: {total:.3f}s  tok/s={PROMPT_COUNT*NEW_TOKENS/total:.2f}")
            continue
        if isinstance(param, int):
            output, st = last
            row = {"label": label, "seconds": total, "speedup": baseline_total / total,
                   "acceptance": st.acceptance_rate, "target_calls": st.target_decode_calls,
                   "exact": int(output == autoregressive_generate(target, prompts[-1], NEW_TOKENS, mode="greedy")[0])}
        else:
            output, st = last
            row = {"label": label, "seconds": total, "speedup": baseline_total / total,
                   "acceptance": st["accepted"] / max(1, st["proposed"]),
                   "target_calls": st["target_calls"],
                   "proposals_per_round": st["proposed"] / st["rounds"],
                   "exact": int(output == autoregressive_generate(target, prompts[-1], NEW_TOKENS, mode="greedy")[0])}
        results.append(row)
        extra = f" proposals/round={row.get('proposals_per_round', K_MAX):.1f}" if "proposals_per_round" in row else ""
        print(f"{label:>16}: {total:.3f}s  speedup={row['speedup']:.3f}x  "
              f"accept={row['acceptance']:.2%} calls={row['target_calls']} exact={row['exact']}{extra}")

    # 全部 prompt 的严格等价校验（对每个 adaptive 配置）
    print("\n严格等价校验（6 prompt 全量）:")
    for theta in THRESHOLDS:
        ok = all(adaptive_speculative_generate(target, draft, p, NEW_TOKENS, K_MAX, theta)[0]
                 == autoregressive_generate(target, p, NEW_TOKENS, mode="greedy")[0] for p in prompts)
        print(f"  theta={theta}: {'PASS' if ok else 'FAIL'}")
        for row in results:
            if row["label"] == f"adaptive_t{theta}":
                row["exact_all_prompts"] = bool(ok)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "adaptive_k.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 图
    labels = [r["label"] for r in results if r["label"] != "baseline"]
    speeds = [r["speedup"] for r in results if r["label"] != "baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#888"] * len(FIXED_KS) + ["#c44"] * len(THRESHOLDS)
    ax.bar(labels, speeds, color=colors)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="target-only baseline")
    best_fixed = max(r["speedup"] for r in results if r["label"].startswith("fixed"))
    ax.axhline(best_fixed, color="#888", lw=0.8, ls=":", label=f"best fixed K ({best_fixed:.2f}x)")
    for i, s in enumerate(speeds):
        ax.text(i, s + 0.02, f"{s:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("wall-clock speedup")
    ax.set_title(f"adaptive K (confidence early-exit, K_max={K_MAX}) vs fixed K")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "adaptive_k.png", dpi=150)
    print(f"\nsaved: {OUT_DIR / 'adaptive_k.json'}\nsaved: {OUT_DIR / 'adaptive_k.png'}")


if __name__ == "__main__":
    main()
