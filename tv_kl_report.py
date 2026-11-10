"""Distillation alignment metrics, corrected for the greedy/stochastic split.

v1 compared teacher-forced sum min(p,q) = 1 - TV(p,q) with *greedy* acceptance
rates from benchmark.json. That is a category error:

* stochastic mode: E[accept] = sum min(p,q) = 1 - TV(p,q) >= 1 - sqrt(KL/2);
* greedy mode: acceptance is ``argmax(q) == argmax(p)``, i.e. top-1 agreement.

v1 also stated that sum min(p,q) is "closer to the acceptance rate than KL"
without computing any correlation or regression, and "measured rates are below
the bound" is a tautology rather than a validation.

v2 measures both modes separately, sweeps the decoding temperature to obtain
several independent conditions, and reports the correlation of each teacher-
forced metric with the *matching* measured quantity. The teacher-forced versus
free-running exposure-bias gap is reported explicitly instead of being hidden.

Outputs: analysis/distill_metrics_v2.{json,md,png}
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))

from specforge.data import TokenCorpus  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.distill_tv import alignment_metrics_extended  # noqa: E402
from specforge.stats import verified_acceptance_rate  # noqa: E402
from specforge.training import load_artifacts  # noqa: E402

ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
OUT_DIR = ROOT / "analysis"
TEMPERATURES = (0.7, 0.9, 1.0, 1.2, 1.5)
NEW_TOKENS = 40
PROMPT_COUNT = 6
DRAFT_K = 8
SAMPLE_REPEATS = 3


def pearson(x, y) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.inference_mode()
def measured_acceptance(target, draft, prompts, temperature: float, mode: str):
    rates, exact = [], True
    for i, prompt in enumerate(prompts):
        repeats = 1 if mode == "greedy" else SAMPLE_REPEATS
        for rep in range(repeats):
            trace: list = []
            out, _ = speculative_generate(
                target, draft, prompt, NEW_TOKENS, draft_k=DRAFT_K,
                mode=mode, temperature=temperature, seed=7000 + 100 * i + rep,
                trace=trace,
            )
            rates.append(verified_acceptance_rate(trace))
            if mode == "greedy":
                base, _ = autoregressive_generate(target, prompt, NEW_TOKENS, mode="greedy")
                exact = exact and (out == base)
    return statistics.fmean(rates), exact


def main() -> None:
    torch.set_num_threads(1)
    target, draft, _, _ = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820)
    prompts = corpus.validation_prompts(count=PROMPT_COUNT, prompt_len=18)

    rows = []
    greedy_rate, greedy_exact = measured_acceptance(target, draft, prompts, 1.0, "greedy")
    print(f"greedy acceptance = {greedy_rate:.4f}, exact = {greedy_exact}")
    for temperature in TEMPERATURES:
        gen = torch.Generator().manual_seed(20261109)
        align = alignment_metrics_extended(
            target, draft, corpus, batch_size=8, seq_len=96, generator=gen,
            temperatures=(temperature,), batches=8,
        )[temperature]
        stochastic, _ = measured_acceptance(target, draft, prompts, temperature, "sample")
        rows.append({
            "temperature": temperature,
            "sum_min_p_q": align["sum_min_p_q"],
            "tv": align["tv"],
            "kl": align["kl_target_to_draft"],
            "top1": align["top1_agreement"],
            "pinsker_lower_bound": align["pinsker_acceptance_lower_bound"],
            "stochastic_acceptance": stochastic,
            "overlap_gap": align["sum_min_p_q"] - stochastic,
            "greedy_acceptance": greedy_rate,
            "top1_gap": align["top1_agreement"] - greedy_rate,
        })
        print(f"T={temperature}: overlap={align['sum_min_p_q']:.4f} "
              f"KL={align['kl_target_to_draft']:.4f} top1={align['top1_agreement']:.4f} "
              f"stoch={stochastic:.4f} gap={rows[-1]['overlap_gap']:+.4f}")

    corr_overlap = pearson([r["sum_min_p_q"] for r in rows],
                           [r["stochastic_acceptance"] for r in rows])
    corr_kl = pearson([r["kl"] for r in rows], [r["stochastic_acceptance"] for r in rows])
    corr_top1 = pearson([r["top1"] for r in rows], [r["greedy_acceptance"] for r in rows])
    pinsker_ok = all(r["stochastic_acceptance"] >= r["pinsker_lower_bound"] - 1e-9 for r in rows)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "distill_metrics_v2.json").write_text(json.dumps({
        "rows": rows,
        "greedy_exact_match": greedy_exact,
        "correlations": {
            "sum_min_vs_stochastic_acceptance": corr_overlap,
            "kl_vs_stochastic_acceptance": corr_kl,
            "top1_vs_greedy_acceptance": corr_top1,
        },
        "pinsker_bound_holds": pinsker_ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    temps = [r["temperature"] for r in rows]
    ax = axes[0]
    ax.plot(temps, [r["sum_min_p_q"] for r in rows], "o-", color="#267", label="sum min(p,q)")
    ax.plot(temps, [r["stochastic_acceptance"] for r in rows], "s-", color="#c44",
            label="measured stochastic acceptance")
    ax.plot(temps, [r["pinsker_lower_bound"] for r in rows], "^--", color="#888",
            label="Pinsker lower bound")
    ax.set_xlabel("temperature")
    ax.set_title("stochastic: theory vs measurement")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(temps, [r["top1"] for r in rows], "o-", color="#267", label="teacher-forced top-1")
    ax.axhline(greedy_rate, color="#c44", lw=2, label="measured greedy acceptance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("temperature")
    ax.set_title("greedy: top-1 agreement is the right predictor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(temps, [r["overlap_gap"] for r in rows], "o-", color="#e90", label="overlap - measured")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("temperature")
    ax.set_ylabel("exposure-bias gap")
    ax.set_title("teacher-forced vs free-running")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distill_metrics_v2.png", dpi=150)
    plt.close(fig)

    lines = [
        "# 蒸馏对齐指标 v2：区分贪婪与随机模式", "",
        "## v1 的错误", "",
        "- `sum min(p,q) = 1 - TV(p,q)` 是**随机采样模式**的接受概率上界，",
        "  与贪婪接受率不是同一个量。贪婪接受事件是 `argmax(q) == argmax(p)`。",
        "- `Pinsker: TV <= sqrt(KL/2)` 给出 `E[accept] >= 1 - sqrt(KL/2)`，",
        "  这才把 KL 蒸馏损失与接受率连成可推导的因果链。",
        "- 「实测均低于上界」是恒真陈述，不构成验证。v1 标题承诺的",
        "  「sum min 比 KL 更贴近接受率」没有任何相关系数或回归支撑。", "",
        f"协议：teacher-forced 对齐在验证集上测量；实测接受率由主库 trace 的风险集口径给出；",
        f"{len(TEMPERATURES)} 个温度形成独立条件用于相关性分析；"
        f"每条件 {SAMPLE_REPEATS} 次采样重复。", "",
        "| T | sum min(p,q) | TV | KL | top-1 | Pinsker 下界 | 随机接受率 | overlap 差距 | 贪婪接受率 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['temperature']} | {r['sum_min_p_q']:.4f} | {r['tv']:.4f} "
            f"| {r['kl']:.4f} | {r['top1']:.4f} | {r['pinsker_lower_bound']:.4f} "
            f"| {r['stochastic_acceptance']:.4f} | {r['overlap_gap']:+.4f} "
            f"| {r['greedy_acceptance']:.4f} |"
        )
    lines += ["", "## 相关性（跨温度）", "",
              f"- corr(sum min(p,q), 随机接受率) = {corr_overlap:+.3f}",
              f"- corr(KL, 随机接受率) = {corr_kl:+.3f}（预期为负）",
              f"- corr(top-1, 贪婪接受率) = {corr_top1:+.3f}（贪婪接受率跨温度为常数，"
              "相关系数不适用，故直接对比水平线）", "",
              "## 结论", "",
              f"1. Pinsker 下界在全部 {len(rows)} 个温度上{'成立' if pinsker_ok else '未完全成立'}。",
              "2. `sum min(p,q)` 是 teacher-forced 分布上的量，自由生成时存在 exposure bias，",
              "   `overlap 差距` 列给出量化结果，不能把两者当成同一指标。",
              "3. 贪婪模式应使用 top-1 一致率解释，与 TV/KL 无直接对应。",
              f"4. 贪婪输出与 target-only 逐 token 等价: {greedy_exact}。", "",
              "![distill metrics v2](distill_metrics_v2.png)", ""]
    (OUT_DIR / "distill_metrics_v2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'distill_metrics_v2.json'}")
    print(f"saved: {OUT_DIR / 'distill_metrics_v2.md'}")
    print(f"saved: {OUT_DIR / 'distill_metrics_v2.png'}")


if __name__ == "__main__":
    main()
