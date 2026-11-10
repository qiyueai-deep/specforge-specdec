"""End-to-end statistical equivalence and stochastic-mode performance audit.

Replaces the v1 claim "rejection-sampling identity holds with error < 1e-7"
(which only checks an algebraic identity on synthetic vectors) with a test on
sequences actually produced by the two decoders.

Outputs: analysis/equivalence.{json,md,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))

from specforge.data import TokenCorpus  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.equivalence import (  # noqa: E402
    ngram_counter,
    run_equivalence_suite,
    sample_sequences,
)
from specforge.training import load_artifacts  # noqa: E402

ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
OUT_DIR = ROOT / "analysis"
CONFIGS = (
    {"label": "T=1.0", "temperature": 1.0, "top_k": None},
    {"label": "T=0.8, top_k=10", "temperature": 0.8, "top_k": 10},
)
N_SAMPLES = 400
N_TOKENS = 24
DRAFT_K = 8


def make_plot(results, details, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    ax = axes[0]
    a, b = details["T=1.0"]
    ca, cb = ngram_counter(a, 1), ngram_counter(b, 1)
    keys = [k for k, _ in ca.most_common(30)]
    xa = np.array([ca[k] for k in keys], dtype=float)
    xb = np.array([cb[k] for k in keys], dtype=float)
    xa /= max(1.0, sum(ca.values()))
    xb /= max(1.0, sum(cb.values()))
    xs = np.arange(len(keys))
    ax.bar(xs - 0.2, xa, width=0.4, label="target-only", color="#267")
    ax.bar(xs + 0.2, xb, width=0.4, label="speculative", color="#c44")
    ax.set_title("top-30 token frequencies")
    ax.set_xlabel("token rank")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    first = results[0]
    null = np.array(details["null_tv"])
    ax.hist(null, bins=30, color="#888", alpha=0.8, label="H0 null TV")
    ax.axvline(first["observed_tv_unigram"], color="#c44", lw=2, label="observed TV")
    ax.axvline(first["null_tv_critical"], color="k", ls="--", label="95% critical value")
    ax.set_title("parametric bootstrap under H0")
    ax.set_xlabel("total variation")
    ax.legend(fontsize=8)

    ax = axes[2]
    effects = [c["effect"] for c in first["power_curve"]]
    powers = [c["power"] for c in first["power_curve"]]
    ax.plot(effects, powers, "o-", color="#267")
    ax.axhline(0.8, color="k", ls=":", label="0.8 power")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("true TV effect size")
    ax.set_ylabel("detection power")
    ax.set_title("test sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import torch

    torch.set_num_threads(1)
    target, draft, tokenizer, _ = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820)
    prompt = corpus.validation_prompts(count=1, prompt_len=18)[0]

    results, details = [], {}
    for cfg in CONFIGS:
        print(f"running {cfg['label']} ...", flush=True)
        result = run_equivalence_suite(
            target, draft, prompt, tokenizer.vocab_size,
            n_samples=N_SAMPLES, n_tokens=N_TOKENS, draft_k=DRAFT_K,
            temperature=cfg["temperature"], top_k=cfg["top_k"], seed=0,
        )
        item = result.to_dict()
        item["label"] = cfg["label"]
        item["top_k"] = cfg["top_k"]
        results.append(item)
        a, b = sample_sequences(
            target, draft, prompt, 200, N_TOKENS, DRAFT_K,
            temperature=cfg["temperature"], top_k=cfg["top_k"], seed_start=10000,
        )
        details[cfg["label"]] = (a, b)
        # Recreate a null TV sample for the histogram.
        from specforge.equivalence import ngram_counter as _nc
        ca, cb = _nc(a, 1), _nc(b, 1)
        va = np.zeros(tokenizer.vocab_size)
        vb = np.zeros(tokenizer.vocab_size)
        for k, v in ca.items():
            if 0 <= int(k) < tokenizer.vocab_size:
                va[int(k)] = v
        for k, v in cb.items():
            if 0 <= int(k) < tokenizer.vocab_size:
                vb[int(k)] = v
        pooled = (va + vb) / max(1, va.sum() + vb.sum())
        rng = np.random.default_rng(3)
        details.setdefault("null_tv", np.array([
            0.5 * np.abs(rng.multinomial(int(va.sum()), pooled) / va.sum()
                         - rng.multinomial(int(vb.sum()), pooled) / vb.sum()).sum()
            for _ in range(2000)
        ]))
        print(f"  TV={item['observed_tv_unigram']:.4f} CI={item['tv_ci_unigram']} "
              f"p={item['p_value']:.3f} reject={item['reject_h0']} "
              f"speedup={item['stochastic_speedup']:.3f}x exact={item['exact_greedy']}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "equivalence.json").write_text(
        json.dumps({"results": results, "protocol": {
            "n_samples": N_SAMPLES, "n_tokens": N_TOKENS, "draft_k": DRAFT_K,
            "test": "parametric bootstrap under pooled multinomial H0",
            "note": "sequences are independent draws, not paired by seed",
        }}, ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(results, details, OUT_DIR / "equivalence.png")

    lines = [
        "# 随机模式输出分布等价性与性能", "",
        "## v1 的缺口", "",
        "v1 只验证 `min(p,q) + (1-sum min(p,q))*normalize(max(p-q,0)) == p`",
        "在随机生成的概率向量上成立，误差 < 1e-7。这是**一步代数恒等式**，",
        "不能说明解码器整体采样自目标分布。且 v1 完全没有随机模式的性能数据。", "",
        "## 检验设计", "",
        f"固定 prompt，target-only 与 speculative 各独立采 {N_SAMPLES} 条长度 {N_TOKENS} 的序列。",
        "H0: 两组来自同一个分类分布。以合并经验分布做参数自助，",
        "得到 TV 的零分布与 p 值；按整条序列自助重采样给出 TV 的 95% CI；",
        "再对给定效应量 delta 计算检出功效（power），使阴性结论可解释。",
        "两组序列不以 seed 配对，比较的是分布而非逐条文本。", "",
        "| 配置 | TV(unigram) | 95% CI | 零分布临界值 | p | 拒绝 H0 | TV(bigram) | 贪婪严格等价 | 随机模式 speedup |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        ci = r["tv_ci_unigram"]
        lines.append(
            f"| {r['label']} | {r['observed_tv_unigram']:.4f} "
            f"| [{ci[0]:.4f}, {ci[1]:.4f}] | {r['null_tv_critical']:.4f} "
            f"| {r['p_value']:.3f} | {r['reject_h0']} | {r['observed_tv_bigram']:.4f} "
            f"| {r['exact_greedy']} | {r['stochastic_speedup']:.3f}x |"
        )
    lines += ["", "## 检出功效", "",
              "| 真实 TV 效应 | power |", "|---:|---:|"]
    for c in results[0]["power_curve"]:
        lines.append(f"| {c['effect']:.2f} | {c['power']:.3f} |")
    lines += ["", "## 结论边界", "",
              "未拒绝 H0 只能结合 power 解读：在小效应量下检验可能没有分辨力，",
              "不能表述为「已证明分布完全相同」。随机模式 speedup 为单 prompt 的",
              "配对测量结果，样本量小，仅用于填补 v1 的性能空白，不作为简历主指标。", "",
              "![equivalence](equivalence.png)", ""]
    (OUT_DIR / "equivalence.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'equivalence.json'}")
    print(f"saved: {OUT_DIR / 'equivalence.md'}")
    print(f"saved: {OUT_DIR / 'equivalence.png'}")


if __name__ == "__main__":
    main()
