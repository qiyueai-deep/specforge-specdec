"""Corpus-entropy stress test: does the 2.34x result survive higher entropy?

The original template corpus has bigram conditional entropy 0.64 nats and a
target validation CE of 0.11, i.e. the next token is nearly deterministic. An
acceptance rate above 90% is then a property of the data, not of the decoder.

This script trains a small target/draft pair on the high-entropy ``rich``
corpus and reports the same acceptance and speedup metrics, so the boundary of
the original claim becomes measurable. It also checks Pinsker's inequality
as a link between teacher-forced KL and measured stochastic acceptance.

Outputs: analysis/entropy_stress.{json,md,png}
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

from specforge.config import ModelConfig, TrainConfig  # noqa: E402
from specforge.data import TokenCorpus, corpus_entropy_report  # noqa: E402
from specforge.decoding import autoregressive_generate, speculative_generate  # noqa: E402
from specforge.distill_tv import alignment_metrics_extended  # noqa: E402
from specforge.model import TinyTransformerLM  # noqa: E402
from specforge.stats import verified_acceptance_rate, wilson_interval  # noqa: E402
from specforge.training import _distill, _train_ce, load_artifacts  # noqa: E402
from specforge.utils import set_seed  # noqa: E402

OUT_DIR = ROOT / "analysis"
TEMPLATE_ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
RICH_STEPS = (140, 100, 140)
NEW_TOKENS = 48
PROMPT_LEN = 18
PROMPT_COUNT = 6
K_VALUES = (4, 8, 16, 28)
REPEATS = 5


def train_rich(corpus):
    set_seed(20261109)
    torch.set_num_threads(1)
    vocab = corpus.tokenizer.vocab_size
    target = TinyTransformerLM(ModelConfig(vocab_size=vocab, max_seq_len=256,
                                           d_model=256, n_layers=6, n_heads=8, d_ff=1024))
    draft = TinyTransformerLM(ModelConfig(vocab_size=vocab, max_seq_len=256,
                                          d_model=80, n_layers=2, n_heads=4, d_ff=240))
    cfg = TrainConfig(
        seed=20261109, seq_len=96, batch_size=12,
        target_steps=RICH_STEPS[0], draft_sft_steps=RICH_STEPS[1],
        distill_steps=RICH_STEPS[2], temperature=2.0, distill_alpha=0.75,
        two_stage_distillation=False, ce_refinement_fraction=0.0,
        eval_batches=4, torch_threads=1, distill_objective="kl_ce",
        corpus_mode="rich",
    )
    gen = torch.Generator().manual_seed(20261109 + 1000)
    started = time.perf_counter()
    _train_ce(target, corpus, cfg, RICH_STEPS[0], gen, "target")
    _train_ce(draft, corpus, cfg, RICH_STEPS[1], gen, "draft")
    _distill(target, draft, corpus, cfg, gen)
    seconds = time.perf_counter() - started
    target.eval()
    draft.eval()
    return target, draft, seconds


def target_ce(model, corpus, batches=6):
    from torch.nn import functional as F

    gen = torch.Generator().manual_seed(999)
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = corpus.batch("validation", 8, 96, gen)
            logits = model(x)
            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
    return statistics.fmean(losses)


def acceptance(target, draft, corpus, k=8, mode="greedy"):
    rates, exact = [], True
    prompts = corpus.validation_prompts(count=PROMPT_COUNT, prompt_len=PROMPT_LEN)
    for i, prompt in enumerate(prompts):
        trace: list = []
        out, _ = speculative_generate(target, draft, prompt, NEW_TOKENS, draft_k=k,
                                      mode=mode, temperature=1.0, seed=3000 + i, trace=trace)
        rates.append(verified_acceptance_rate(trace))
        if mode == "greedy":
            base, _ = autoregressive_generate(target, prompt, NEW_TOKENS, mode="greedy")
            exact = exact and (out == base)
    return statistics.fmean(rates), exact


def speedups(target, draft, corpus):
    prompts = corpus.validation_prompts(count=PROMPT_COUNT, prompt_len=PROMPT_LEN)
    configs = {"baseline": lambda p: autoregressive_generate(target, p, NEW_TOKENS, mode="greedy")}
    for k in K_VALUES:
        configs[f"k{k}"] = lambda p, k=k: speculative_generate(
            target, draft, p, NEW_TOKENS, draft_k=k, mode="greedy")
    labels = list(configs)
    durations = {lab: [[] for _ in prompts] for lab in labels}
    for lab in labels:
        for p in prompts:
            configs[lab](p)
    for rep in range(REPEATS):
        order = labels[rep % len(labels):] + labels[:rep % len(labels)]
        for i, p in enumerate(prompts):
            for lab in order:
                t0 = time.perf_counter()
                configs[lab](p)
                durations[lab][i].append(time.perf_counter() - t0)
    totals = {lab: sum(statistics.median(x) for x in durations[lab]) for lab in labels}
    return {f"k{k}": totals["baseline"] / totals[f"k{k}"] for k in K_VALUES}


def main() -> None:
    torch.set_num_threads(1)
    template_corpus = TokenCorpus.create(20260820, mode="template")
    rich_corpus = TokenCorpus.create(20260820, mode="rich")
    entropy = {
        "template": corpus_entropy_report(template_corpus.validation_ids, template_corpus.tokenizer),
        "rich": corpus_entropy_report(rich_corpus.validation_ids, rich_corpus.tokenizer),
    }

    print("training rich-corpus models ...", flush=True)
    rich_target, rich_draft, train_seconds = train_rich(rich_corpus)
    print(f"rich training took {train_seconds:.1f}s", flush=True)

    t_target, t_draft, t_tok, _ = load_artifacts(TEMPLATE_ARTIFACTS)
    gen = torch.Generator().manual_seed(555)
    align = {
        "template": alignment_metrics_extended(t_target, t_draft, template_corpus,
                                                8, 96, gen, temperatures=(1.0,), batches=6)[1.0],
        "rich": alignment_metrics_extended(rich_target, rich_draft, rich_corpus,
                                            8, 96, gen, temperatures=(1.0,), batches=6)[1.0],
    }
    rows = {}
    for name, corpus, target, draft in (
        ("template", template_corpus, t_target, t_draft),
        ("rich", rich_corpus, rich_target, rich_draft),
    ):
        greedy, exact = acceptance(target, draft, corpus, k=8, mode="greedy")
        stochastic, _ = acceptance(target, draft, corpus, k=8, mode="sample")
        rows[name] = {
            "entropy": entropy[name],
            "vocab_size": corpus.tokenizer.vocab_size,
            "target_validation_ce": target_ce(target, corpus),
            "alignment_t1": align[name],
            "greedy_acceptance": greedy,
            "greedy_exact_match": exact,
            "stochastic_acceptance": stochastic,
            "pinsker_lower_bound": align[name]["pinsker_acceptance_lower_bound"],
            "speedups": speedups(target, draft, corpus),
        }
        print(f"{name}: H2={entropy[name]['bigram_conditional_entropy_nats']:.3f} "
              f"CE={rows[name]['target_validation_ce']:.3f} "
              f"greedy={greedy:.4f} stoch={stochastic:.4f} "
              f"best_speedup={max(rows[name]['speedups'].values()):.3f}x", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "entropy_stress.json").write_text(
        json.dumps({"rows": rows, "rich_train_seconds": train_seconds,
                    "rich_steps": RICH_STEPS}, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    names = ["template", "rich"]
    ax = axes[0]
    ax.bar(names, [rows[n]["entropy"]["bigram_conditional_entropy_nats"] for n in names],
           color=["#267", "#c44"])
    ax.set_ylabel("nats")
    ax.set_title("bigram conditional entropy")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    xs = [0, 1]
    ax.bar([x - 0.18 for x in xs], [rows[n]["greedy_acceptance"] for n in names],
           width=0.35, label="greedy", color="#267")
    ax.bar([x + 0.18 for x in xs], [rows[n]["stochastic_acceptance"] for n in names],
           width=0.35, label="stochastic", color="#c44")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("risk-set acceptance rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ks = list(K_VALUES)
    for name, color in (("template", "#267"), ("rich", "#c44")):
        ax.plot(ks, [rows[name]["speedups"][f"k{k}"] for k in ks], "o-", color=color, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("wall-clock speedup")
    ax.set_title("speedup under different entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "entropy_stress.png", dpi=150)
    plt.close(fig)

    lines = [
        "# 语料熵压力测试", "",
        "原语料是 8 类固定句式，二元条件熵仅 "
        f"{rows['template']['entropy']['bigram_conditional_entropy_nats']:.3f} nats，"
        f"target 验证 CE {rows['template']['target_validation_ce']:.3f}，",
        "下一 token 近乎确定。高熵语料是稀疏马尔可夫链，用于检验算法结论的边界。", "",
        "| 语料 | 词表 | H1 | H2 | 可预测性 | target CE | sum min(p,q) | KL | Pinsker 下界 | 贪婪接受率 | 随机接受率 | 贪婪等价 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for n in names:
        r = rows[n]
        e, a = r["entropy"], r["alignment_t1"]
        lines.append(
            f"| {n} | {r['vocab_size']} | {e['unigram_entropy_nats']:.3f} "
            f"| {e['bigram_conditional_entropy_nats']:.3f} | {e['predictability']:.3f} "
            f"| {r['target_validation_ce']:.3f} | {a['sum_min_p_q']:.4f} "
            f"| {a['kl_target_to_draft']:.4f} | {r['pinsker_lower_bound']:.4f} "
            f"| {r['greedy_acceptance']:.4f} | {r['stochastic_acceptance']:.4f} "
            f"| {r['greedy_exact_match']} |"
        )
    lines += ["", "## 各 K 的墙钟加速", "",
              "| 语料 | " + " | ".join(f"K={k}" for k in K_VALUES) + " |",
              "|---|" + "---:|" * len(K_VALUES)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(
            f"{rows[n]['speedups'][f'k{k}']:.3f}x" for k in K_VALUES) + " |")
    lines += ["", "## 结论", "",
              f"高熵语料把 H2 从 {rows['template']['entropy']['bigram_conditional_entropy_nats']:.3f} "
              f"提升到 {rows['rich']['entropy']['bigram_conditional_entropy_nats']:.3f} nats，",
              f"贪婪接受率从 {rows['template']['greedy_acceptance']:.4f} 变为 "
              f"{rows['rich']['greedy_acceptance']:.4f}，"
              f"最优加速从 {max(rows['template']['speedups'].values()):.3f}x 变为 "
              f"{max(rows['rich']['speedups'].values()):.3f}x。",
              "这说明原项目报告的高接受率与加速比高度依赖低熵语料，",
              "简历与报告必须保留该边界，不得外推到真实模型。", "",
              "![entropy stress](entropy_stress.png)", ""]
    (OUT_DIR / "entropy_stress.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {OUT_DIR / 'entropy_stress.json'}")
    print(f"saved: {OUT_DIR / 'entropy_stress.md'}")
    print(f"saved: {OUT_DIR / 'entropy_stress.png'}")


if __name__ == "__main__":
    main()
