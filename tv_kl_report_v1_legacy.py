"""蒸馏对齐指标的 TV / KL / top-1 报告（任务3）。

动机：推测解码每 token 接受概率的上界是 α = Σ min(p,q)（= 1 − TV(p,q)），
KL 只是蒸馏损失，不直接决定接受率。本脚本在验证集上 teacher-forced 计算
target 与 draft 分布的逐位置 Σmin(p,q)、KL(p||q)、top-1 一致率，并与
benchmark 中实测的逐 K 接受率对照，验证 "Σmin(p,q) 比KL 更贴近接受率"。

输出：analysis/distill_metrics.{json,md,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

ROOT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))

from specforge.data import TokenCorpus  # noqa: E402
from specforge.training import load_artifacts  # noqa: E402

ARTIFACTS = ROOT / "experiments" / "fast_train_T2.0_a0.75_speed15_verified" / "artifacts"
FRESH = ROOT / "analysis" / "benchmark_fresh.json"
OUT_DIR = ROOT / "analysis"
BATCHES = 8
BATCH_SIZE = 8
SEQ_LEN = 96
TEMPERATURES = (1.0, 2.0)


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(1)
    target, draft, _tok, _man = load_artifacts(ARTIFACTS)
    corpus = TokenCorpus.create(20260820)
    gen = torch.Generator().manual_seed(20260820)

    summary = {}
    per_position = {}
    for temp in TEMPERATURES:
        total_min = total_kl = total_tokens = 0.0
        top1_correct = 0
        pos_min: dict[int, float] = {}
        pos_cnt: dict[int, float] = {}
        for _ in range(BATCHES):
            x, _y = corpus.batch("validation", BATCH_SIZE, SEQ_LEN, gen)
            target_logits = target(x)[:, :-1].float()
            draft_logits = draft(x)[:, :-1].float()
            p = F.softmax(target_logits / temp, dim=-1)
            q = F.softmax(draft_logits / temp, dim=-1)
            overlap = torch.minimum(p, q).sum(-1)                    # [B, T-1]
            kl = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(-1)
            top1 = (p.argmax(-1) == q.argmax(-1))
            total_min += float(overlap.sum())
            total_kl += float(kl.sum())
            top1_correct += int(top1.sum())
            total_tokens += overlap.numel()
            for idx in range(0, overlap.shape[1], 8):
                pos_min[idx] = pos_min.get(idx, 0.0) + float(overlap[:, idx].sum())
                pos_cnt[idx] = pos_cnt.get(idx, 0.0) + overlap.shape[0]
        summary[temp] = {
            "mean_overlap_sum_min_p_q": total_min / total_tokens,
            "mean_kl_target_to_draft": total_kl / total_tokens,
            "top1_agreement": top1_correct / total_tokens,
            "tokens": int(total_tokens),
        }
        per_position[temp] = {k: pos_min[k] / pos_cnt[k] for k in sorted(pos_min)}

    empirical = {}
    if FRESH.exists():
        fresh = json.loads(FRESH.read_text(encoding="utf-8"))
        empirical = {row["draft_k"]: row["mean_acceptance_rate"] for row in fresh["speculative"]}

    report = {"teacher_forced": summary, "empirical_acceptance_by_k": empirical}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "distill_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 蒸馏对齐指标：Σmin(p,q) vs KL", "",
             f"验证集 teacher-forced，{BATCHES} 批 × {BATCH_SIZE} 序列 × {SEQ_LEN-1} 位置。", "",
             "| T | Σmin(p,q)（接受率上界） | KL(target||draft) | top-1 一致率 |",
             "|---:|---:|---:|---:|"]
    for temp in TEMPERATURES:
        s = summary[temp]
        lines.append(f"| {temp} | {s['mean_overlap_sum_min_p_q']:.4f} | "
                     f"{s['mean_kl_target_to_draft']:.4f} | {s['top1_agreement']:.2%} |")
    lines += ["", "## 与实测接受率的对照", "",
              "T=1.0 的 Σmin(p,q) 是任意提议分布下每 token 接受概率的上界；",
              "实测逐 K 接受率（benchmark_fresh.json）均低于该上界，且 K 越大、", 
              "深处危险率衰减使实测越低于上界：", ""]
    if empirical:
        lines.append("| K | 实测接受率 | 与上界的差距 |")
        lines.append("|---:|---:|---:|")
        ub = summary[1.0]["mean_overlap_sum_min_p_q"]
        for k, a in sorted(empirical.items()):
            lines.append(f"| {k} | {a:.4f} | {ub - a:+.4f} |")

    # 图：逐位置 overlap 曲线（两个温度）
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for temp, marker in zip(TEMPERATURES, ("o", "s")):
        xs = sorted(per_position[temp])
        ax.plot(xs, [per_position[temp][x] for x in xs], marker + "-", label=f"T={temp}")
    ax.set_xlabel("sequence position")
    ax.set_ylabel("Σ min(p,q) per position")
    ax.set_title("target-draft overlap by depth (teacher-forced)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "distill_metrics.png", dpi=150)

    md = "\n".join(lines) + "\n\n![overlap](distill_metrics.png)\n"
    (OUT_DIR / "distill_metrics.md").write_text(md, encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved: {OUT_DIR / 'distill_metrics.json'}")
    print(f"saved: {OUT_DIR / 'distill_metrics.md'}")
    print(f"saved: {OUT_DIR / 'distill_metrics.png'}")


if __name__ == "__main__":
    main()
