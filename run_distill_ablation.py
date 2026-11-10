"""Distillation-objective ablation with statistical replication.

Question: KL is only a proxy for the speculative-decoding acceptance rate.
In stochastic mode E[accept] = sum min(p,q) = 1 - TV(p,q), and Pinsker gives
E[accept] >= 1 - sqrt(KL/2). Does directly optimising 1 - sum min(p,q) improve
the *measured* stochastic acceptance rate relative to the original KL+CE loss?

The v1 tv_kl_report.py compared sum min(p,q) against greedy acceptance rates,
which is a category error (greedy acceptance is top-1 agreement). Here greedy
and stochastic acceptance are measured separately, and the comparison is made
against the correct quantity.

Protocol
--------
* The target checkpoint is frozen and shared by every run.
* Only the draft is retrained, with 2 seeds per objective.
* All acceptance rates use the production decoder's risk-set trace.
* Intervals are percentile bootstrap across seeds and sampling repeats.

Outputs: analysis/distill_ablation.{json,md,png}
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

from specforge.config import TrainConfig  # noqa: E402
from specforge.data import TokenCorpus  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.distill_tv import alignment_metrics_extended  # noqa: E402
from specforge.model import TinyTransformerLM  # noqa: E402
from specforge.stats import bootstrap_ci, verified_acceptance_rate  # noqa: E402
from specforge.training import _distill, _train_ce, load_artifacts, load_checkpoint  # noqa: E402
from specforge.utils import set_seed  # noqa: E402

ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
OUT_DIR = ROOT / "analysis"
OBJECTIVES = ("ce_only", "kl_ce", "tv_ce", "kl_tv_ce")
SEEDS = (20260820, 20260831)
SFT_STEPS = 50
DISTILL_STEPS = 90
EVAL_TOKENS = 32
EVAL_K = 8
SAMPLE_REPEATS = 2


def train_one(target, corpus, objective: str, seed: int):
    set_seed(seed)
    reference = load_checkpoint(ARTIFACTS / "draft.pt")
    draft = TinyTransformerLM(reference.config)
    config = TrainConfig(
        seed=seed, seq_len=64, batch_size=8,
        target_steps=0, draft_sft_steps=SFT_STEPS, distill_steps=DISTILL_STEPS,
        learning_rate=2.5e-3, min_learning_rate=2.5e-4, weight_decay=0.01,
        temperature=2.0, distill_alpha=0.75,
        two_stage_distillation=False, ce_refinement_fraction=0.0,
        eval_batches=2, torch_threads=1,
        distill_objective=objective, tv_weight=0.75,
    )
    generator = torch.Generator().manual_seed(seed + 1000)
    started = time.perf_counter()
    _train_ce(draft, corpus, config, SFT_STEPS, generator, "draft")
    _distill(target, draft, corpus, config, generator)
    seconds = time.perf_counter() - started
    draft.eval()
    return draft, seconds


@torch.inference_mode()
def evaluate(target, draft, corpus) -> dict:
    gen = torch.Generator().manual_seed(777)
    alignment = alignment_metrics_extended(
        target, draft, corpus, batch_size=8, seq_len=64, generator=gen,
        temperatures=(1.0, 2.0), batches=4,
    )
    prompts = corpus.validation_prompts(count=4, prompt_len=18)
    greedy_rates, stochastic_rates = [], []
    exact = True
    for prompt in prompts:
        baseline, _ = autoregressive_generate(target, prompt, EVAL_TOKENS, mode="greedy")
        trace: list = []
        output, _ = speculative_generate(target, draft, prompt, EVAL_TOKENS,
                                         draft_k=EVAL_K, mode="greedy", trace=trace)
        exact = exact and (output == baseline)
        greedy_rates.append(verified_acceptance_rate(trace))
        for rep in range(SAMPLE_REPEATS):
            trace = []
            speculative_generate(target, draft, prompt, EVAL_TOKENS,
                                 draft_k=EVAL_K, mode="sample", temperature=1.0,
                                 seed=5000 + rep, trace=trace)
            stochastic_rates.append(verified_acceptance_rate(trace))
    return {
        "alignment_t1": alignment[1.0],
        "alignment_t2": alignment[2.0],
        "greedy_acceptance": statistics.fmean(greedy_rates),
        "stochastic_acceptance": statistics.fmean(stochastic_rates),
        "stochastic_acceptance_ci": list(bootstrap_ci(stochastic_rates, n_boot=1000, seed=9)),
        "top1_agreement_t1": alignment[1.0]["top1_agreement"],
        "greedy_exact_match": exact,
    }


def make_plot(rows, path: Path) -> None:
    labels = [r["objective"] for r in rows]
    metrics = [
        ("sum_min_t1", "sum min(p,q) at T=1", "#267"),
        ("stochastic_acceptance", "measured stochastic acceptance", "#c44"),
        ("greedy_acceptance", "measured greedy acceptance", "#e90"),
        ("kl_t1", "KL(T||S) at T=1", "#639"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (key, title, color) in zip(axes, metrics):
        values = [r[key] for r in rows]
        errs = [r.get(key + "_halfwidth", 0.0) for r in rows]
        ax.bar(labels, values, yerr=errs, capsize=3, color=color, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(1)
    target, _, _, _ = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820, mode="template")

    records = []
    print(f"{'objective':>12} {'seed':>10} {'sec':>7} {'overlap':>9} {'KL':>8} "
          f"{'top1':>7} {'greedy':>8} {'stoch':>8} {'exact':>6}")
    for objective in OBJECTIVES:
        for seed in SEEDS:
            draft, seconds = train_one(target, corpus, objective, seed)
            metrics = evaluate(target, draft, corpus)
            records.append({
                "objective": objective,
                "seed": seed,
                "train_seconds": seconds,
                **metrics,
            })
            print(f"{objective:>12} {seed:>10} {seconds:>7.1f} "
                  f"{metrics['alignment_t1']['sum_min_p_q']:>9.4f} "
                  f"{metrics['alignment_t1']['kl_target_to_draft']:>8.4f} "
                  f"{metrics['top1_agreement_t1']:>7.2%} "
                  f"{metrics['greedy_acceptance']:>8.4f} "
                  f"{metrics['stochastic_acceptance']:>8.4f} "
                  f"{str(metrics['greedy_exact_match']):>6}")

    rows = []
    for objective in OBJECTIVES:
        group = [r for r in records if r["objective"] == objective]
        def mean_of(key):
            return statistics.fmean(r[key] for r in group)
        stoch_samples = []
        for r in group:
            stoch_samples.append(r["stochastic_acceptance"])
        ci = bootstrap_ci(stoch_samples, n_boot=1000, seed=17) if len(group) > 1 else (
            group[0]["stochastic_acceptance"], group[0]["stochastic_acceptance"])
        row = {
            "objective": objective,
            "sum_min_t1": statistics.fmean(r["alignment_t1"]["sum_min_p_q"] for r in group),
            "kl_t1": statistics.fmean(r["alignment_t1"]["kl_target_to_draft"] for r in group),
            "top1_t1": statistics.fmean(r["alignment_t1"]["top1_agreement"] for r in group),
            "pinsker_bound": statistics.fmean(
                r["alignment_t1"]["pinsker_acceptance_lower_bound"] for r in group),
            "greedy_acceptance": mean_of("greedy_acceptance"),
            "stochastic_acceptance": mean_of("stochastic_acceptance"),
            "stochastic_ci": list(ci),
            "stochastic_acceptance_halfwidth": (ci[1] - ci[0]) / 2,
            "greedy_exact_match_all": all(r["greedy_exact_match"] for r in group),
            "train_seconds_mean": mean_of("train_seconds"),
        }
        rows.append(row)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "distill_ablation.json").write_text(
        json.dumps({"rows": rows, "records": records,
                    "protocol": {"objectives": OBJECTIVES, "seeds": SEEDS,
                                 "sft_steps": SFT_STEPS, "distill_steps": DISTILL_STEPS,
                                 "target": "frozen pretrained checkpoint"}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    make_plot(rows, OUT_DIR / "distill_ablation.png")

    best_kl = next(r for r in rows if r["objective"] == "kl_ce")
    best_tv = max((r for r in rows if r["objective"] in ("tv_ce", "kl_tv_ce")),
                  key=lambda r: r["stochastic_acceptance"])
    delta = best_tv["stochastic_acceptance"] - best_kl["stochastic_acceptance"]
    pinsker_ok = all(r["stochastic_acceptance"] >= r["pinsker_bound"] - 1e-9 for r in rows)

    lines = [
        "# 蒸馏目标消融：TV 与 KL 谁更贴近接受率", "",
        "## 修正 v1 的范畴错误", "",
        "随机模式下 `E[accept] = sum min(p,q) = 1 - TV(p,q)`，",
        "Pinsker 不等式给出 `E[accept] >= 1 - sqrt(KL/2)`。",
        "贪婪模式下接受事件是 `argmax(q) == argmax(p)`，由 top-1 一致率决定。",
        "v1 把 `sum min(p,q)` 与贪婪接受率并列比较，属于把两个不同量放在同一张表。",
        "本脚本分别测量两种模式，且都用主库 trace 的风险集口径。", "",
        f"协议：冻结 target，只重训 draft；{len(OBJECTIVES)} 个目标 × {len(SEEDS)} 个 seed；",
        f"SFT {SFT_STEPS} 步 + 蒸馏 {DISTILL_STEPS} 步；教师强制对齐在验证集上测量。", "",
        "## 结果（跨 seed 均值）", "",
        "| 目标 | sum min(p,q) T=1 | KL T=1 | top-1 T=1 | Pinsker 下界 | 贪婪接受率 | 随机接受率 | 随机 95% CI | 贪婪严格等价 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['objective']} | {r['sum_min_t1']:.4f} | {r['kl_t1']:.4f} "
            f"| {r['top1_t1']:.2%} | {r['pinsker_bound']:.4f} "
            f"| {r['greedy_acceptance']:.4f} | {r['stochastic_acceptance']:.4f} "
            f"| [{r['stochastic_ci'][0]:.4f}, {r['stochastic_ci'][1]:.4f}] "
            f"| {r['greedy_exact_match_all']} |"
        )
    lines += [
        "", "## 结论", "",
        f"1. Pinsker 下界在全部 {len(rows)} 个目标上{'成立' if pinsker_ok else '未完全成立'}。",
        f"2. 相对原版 `kl_ce`，最优 TV-aware 目标 `{best_tv['objective']}` 的随机模式接受率"
        f"变化 {delta:+.4f}。",
        "3. `sum min(p,q)` 是 teacher-forced 分布上的量，自由生成时存在 exposure bias，",
        "   因此它与实测随机接受率的差距需要单独报告，不能当成同一指标。",
        "4. 贪婪接受率与 top-1 一致率对应，与 `sum min(p,q)` 不对应。", "",
        "![distillation ablation](distill_ablation.png)", "",
        "> 边界：在低熵模板语料上各目标的分布重叠都很高，差异可能不显著。",
        "> 高熵压力语料见 specforge/data.py 的 rich 模式与 entropy_stress_test.py。", "",
    ]
    (OUT_DIR / "distill_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'distill_ablation.json'}")
    print(f"saved: {OUT_DIR / 'distill_ablation.md'}")
    print(f"saved: {OUT_DIR / 'distill_ablation.png'}")


if __name__ == "__main__":
    main()
