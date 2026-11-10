"""Attribution: where does the wall-clock speedup actually come from?

The v1 report implied that reducing target *calls* causes the 2.34x speedup.
But one 28-token verification forward and one 1-token bonus forward are both
counted as "one call", while their FLOPs differ by an order of magnitude. This
script measures the elementary costs and decomposes the speedup into:

  C_single      one 1-token target forward
  C_block(K)    one K-token target block forward
  amortisation  K * C_single / C_block(K)   (batch matrix-multiply efficiency)
  C_draft       one 1-token draft forward

It also measures a chunked baseline (one forward over the whole generated
block, no incremental cache) to separate Python dispatch overhead from
compute. Finally it tries the torch.compile path, if a CPU compiler exists,
and reports the speedup with dispatch overhead largely removed.

Outputs: analysis/attribution.{json,md,png}
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

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))

from specforge.costs import microbenchmark_components  # noqa: E402
from specforge.data import TokenCorpus  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.training import load_artifacts  # noqa: E402

ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
OUT_DIR = ROOT / "analysis"
NEW_TOKENS = 64
PROMPT_COUNT = 6
K_VALUES = (4, 8, 12, 20, 28, 32)
REPEATS = 7


def timed(fn, repeats=REPEATS):
    for _ in range(1):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


@torch.inference_mode()
def chunked_forward_seconds(target, prompt, n_tokens):
    """One target forward over the entire block, no KV cache, no decode loop."""
    ids = torch.tensor([prompt + [prompt[-1]] * n_tokens], dtype=torch.long)

    def run():
        target(ids)
    return timed(run)


def main() -> None:
    torch.set_num_threads(1)
    target, draft, _, _ = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820)
    prompts = corpus.validation_prompts(count=PROMPT_COUNT, prompt_len=18)

    micro = microbenchmark_components(target, draft, prompts[0], repeats=30)
    c_single = micro["target_single_token_seconds"]
    c_draft = micro["draft_single_token_seconds"]
    blocks = micro["target_block_seconds_by_k"]

    baseline_seconds = sum(
        timed(lambda p=p: autoregressive_generate(target, p, NEW_TOKENS, mode="greedy"))
        for p in prompts
    )
    chunked_seconds = sum(chunked_forward_seconds(target, p, NEW_TOKENS) for p in prompts)

    rows = []
    for k in K_VALUES:
        spec_seconds = sum(
            timed(lambda p=p, k=k: speculative_generate(target, draft, p, NEW_TOKENS,
                                                        draft_k=k, mode="greedy"))
            for p in prompts
        )
        c_block = blocks.get(k, blocks[max(blocks)])
        amortisation = k * c_single / c_block
        # Analytic prediction: rounds * (block + single target forward + draft
        # forwards), versus 64 single target forwards for the baseline.
        rows.append({
            "k": k,
            "speculative_seconds": spec_seconds,
            "measured_speedup": baseline_seconds / spec_seconds,
            "block_forward_ms": c_block * 1000,
            "amortisation_factor": amortisation,
            "predicted_compute_ratio": (NEW_TOKENS * c_single) / max(
                1e-9, (NEW_TOKENS / (k + 1)) * (c_block + c_single + (k + 1) * c_draft)),
        })
        print(f"K={k:>2}: speedup={rows[-1]['measured_speedup']:.3f}x "
              f"block={c_block*1000:.2f}ms amortisation={amortisation:.2f}x")

    dispatch_fraction = 1.0 - chunked_seconds / baseline_seconds
    compile_result = None
    try:
        comp_base = sum(
            timed(lambda p=p: autoregressive_generate(target, p, NEW_TOKENS,
                                                      mode="greedy", use_compile=True))
            for p in prompts
        )
        comp_spec = sum(
            timed(lambda p=p: speculative_generate(target, draft, p, NEW_TOKENS,
                                                   draft_k=28, mode="greedy", use_compile=True))
            for p in prompts
        )
        compile_result = {
            "available": True,
            "baseline_seconds": comp_base,
            "speculative_seconds": comp_spec,
            "speedup": comp_base / comp_spec,
        }
    except Exception as exc:  # torch.compile needs a working CPU C++ compiler
        compile_result = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "micro": micro,
        "baseline_seconds": baseline_seconds,
        "chunked_seconds": chunked_seconds,
        "dispatch_overhead_fraction": dispatch_fraction,
        "rows": rows,
        "compile_control": compile_result,
    }
    (OUT_DIR / "attribution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ks = sorted(blocks)
    ax.plot(ks, [blocks[k] * 1000 for k in ks], "o-", color="#267", label="measured C_block(K)")
    ax.plot(ks, [k * c_single * 1000 for k in ks], "s--", color="#c44",
            label="K x C_single (no batching)")
    ax.set_xlabel("K")
    ax.set_ylabel("ms")
    ax.set_title("target forward cost vs block length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(["baseline", "chunked\n(1 forward)", "spec K=28"],
           [baseline_seconds, chunked_seconds,
            next(r["speculative_seconds"] for r in rows if r["k"] == 28)],
           color=["#267", "#888", "#c44"])
    ax.set_ylabel("seconds, 6 prompts x 64 tokens")
    ax.set_title("dispatch overhead vs speculative decoding")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attribution.png", dpi=150)
    plt.close(fig)

    lines = [
        "# 加速来源归因", "",
        "v1 用「目标调用减少 90.10%」解释 2.34x，但一次 28-token 块验证与",
        "一次 1-token bonus 前向都被记成 1 次调用，成本相差一个量级。",
        "本脚本实测基本前向成本并分解收益。", "",
        f"- 单次 1-token target 前向: {c_single*1000:.3f} ms",
        f"- 单次 1-token draft 前向: {c_draft*1000:.3f} ms",
        f"- 6 prompt x 64 token 的增量自回归基线: {baseline_seconds:.3f} s",
        f"- 同样 6 prompt 的整块单次前向（无解码循环）: {chunked_seconds:.3f} s",
        f"- Python/调度开销占基线比例: {dispatch_fraction:.1%}", "",
        "| K | C_block(K) ms | 批处理摊薄倍数 | 实测 speedup |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['k']} | {r['block_forward_ms']:.3f} "
                     f"| {r['amortisation_factor']:.2f}x | {r['measured_speedup']:.3f}x |")
    lines += ["", "## 结论", ""]
    if dispatch_fraction > 0.5:
        lines.append("基线的相当一部分时间来自逐 token 的调度开销，因此减少调用次数是主要收益来源。")
    else:
        lines.append("整块前向与增量基线耗时接近，说明调度开销不是主因；"
                     "收益主要来自把 K 次 1-token 前向合并成一次 K-token 块前向的矩阵乘摊薄。")
    lines += [
        f"批处理摊薄倍数在 K=28 处约 {next(r['amortisation_factor'] for r in rows if r['k']==28):.2f}x，",
        "这是 CPU 上微型 Transformer 的特性，不能直接外推到 GPU 大模型服务。", "",
        "## torch.compile 对照", "",
    ]
    if compile_result["available"]:
        lines += [
            f"compiled baseline {compile_result['baseline_seconds']:.3f}s，"
            f"compiled speculative {compile_result['speculative_seconds']:.3f}s，"
            f"speedup {compile_result['speedup']:.3f}x。",
            "若编译后加速比明显下降，说明 eager 模式下的收益部分来自 Python 调度。",
        ]
    else:
        lines += [f"本机不可用: {compile_result['reason']}",
                  "该对照需要在带 CPU C++ 编译器的环境重跑。"]
    lines += ["", "![attribution](attribution.png)", ""]
    (OUT_DIR / "attribution.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'attribution.json'}")
    print(f"saved: {OUT_DIR / 'attribution.md'}")
    print(f"saved: {OUT_DIR / 'attribution.png'}")
    print(f"dispatch overhead fraction = {dispatch_fraction:.1%}")


if __name__ == "__main__":
    main()
