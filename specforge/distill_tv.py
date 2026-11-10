"""Distillation objectives that target the acceptance rate directly.

In stochastic speculative decoding the probability that a draft token is
accepted is

    E[accept] = sum_x min(p(x), q(x)) = 1 - TV(p, q),

where p is the target distribution and q the draft distribution, both at the
*decoding* temperature. KL(p||q) is only a proxy: Pinsker's inequality gives
TV <= sqrt(KL/2), hence the guarantee

    E[accept] >= 1 - sqrt(KL(p||q)/2).

So minimising KL controls a lower bound on the acceptance rate, while
minimising 1 - sum min(p,q) optimises the quantity that actually enters the
decoder. This module adds the latter as a differentiable training objective.

Important scope note: in *greedy* decoding the acceptance event is
``argmax(q) == argmax(p)``, which is top-1 agreement, not sum min(p,q). The
v1 report compared sum min(p,q) to greedy acceptance rates, which is a
category error. Use this module's overlap metric against *stochastic* mode
acceptance only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .stats import pinsker_acceptance_lower_bound


def _softmax_t(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits.float() / temperature, dim=-1)


def overlap_and_tv(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean sum min(p,q), mean TV(p,q)) over all positions."""
    with torch.no_grad():
        p = _softmax_t(teacher_logits, temperature)
    q = _softmax_t(student_logits, temperature)
    overlap = torch.minimum(p, q).sum(-1)
    return overlap.mean(), (1.0 - overlap).mean()


def kl_teacher_student(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    seq_len: int,
) -> torch.Tensor:
    """The same temperature-scaled KD term used by the original training loop."""
    with torch.no_grad():
        teacher_prob = _softmax_t(teacher_logits, temperature)
    student_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") / seq_len


def build_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    alpha: float,
    temperature: float,
    tv_weight: float = 0.75,
    tv_scale: float | None = None,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Dispatch a distillation objective.

    ``kl_ce`` is bit-for-bit the original loss, so the default training run is
    unchanged. ``tv_ce`` and ``kl_tv_ce`` are the new TV-aware objectives.
    ``tv_scale`` compensates for the different magnitude of TV and CE; it
    defaults to temperature**2, matching the classic KD gradient rescaling.
    """
    n = seq_len if seq_len is not None else student_logits.shape[1]
    scale = tv_scale if tv_scale is not None else temperature ** 2
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]).float(),
        labels.reshape(-1),
    )

    if objective == "ce_only":
        return ce
    if objective == "kl_ce":
        kd = kl_teacher_student(student_logits, teacher_logits, temperature, n)
        return alpha * (temperature ** 2) * kd + (1.0 - alpha) * ce
    if objective == "tv_only":
        overlap, _ = overlap_and_tv(student_logits, teacher_logits, temperature)
        return scale * (1.0 - overlap)
    if objective == "tv_ce":
        overlap, _ = overlap_and_tv(student_logits, teacher_logits, temperature)
        return tv_weight * scale * (1.0 - overlap) + (1.0 - tv_weight) * ce
    if objective == "kl_tv_ce":
        kd = kl_teacher_student(student_logits, teacher_logits, temperature, n)
        overlap, _ = overlap_and_tv(student_logits, teacher_logits, temperature)
        half = tv_weight / 2.0
        return (alpha * (temperature ** 2) * kd
                + half * scale * (1.0 - overlap)
                + (1.0 - alpha - half) * ce)
    raise ValueError(f"unknown distillation objective: {objective}")


@dataclass
class AlignmentRow:
    temperature: float
    overlap: float
    tv: float
    kl: float
    top1: float
    pinsker_lower_bound: float

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "sum_min_p_q": self.overlap,
            "tv": self.tv,
            "kl_target_to_draft": self.kl,
            "top1_agreement": self.top1,
            "pinsker_acceptance_lower_bound": self.pinsker_lower_bound,
        }


@torch.no_grad()
def alignment_metrics_extended(
    target,
    draft,
    corpus,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
    temperatures=(1.0, 2.0),
    batches: int = 8,
) -> dict:
    """Teacher-forced alignment at several temperatures.

    Returns sum min(p,q), TV, KL and top-1 agreement, plus the Pinsker lower
    bound on the stochastic-mode acceptance rate derived from the measured KL.
    """
    target.eval()
    draft.eval()
    rows: dict[float, dict] = {}
    for temperature in temperatures:
        overlap_sum = tv_sum = kl_sum = 0.0
        top1_hits = tokens = 0
        for _ in range(batches):
            x, _y = corpus.batch("validation", batch_size, seq_len, generator)
            p = _softmax_t(target(x), temperature)
            student = draft(x)
            q = _softmax_t(student, temperature)
            overlap = torch.minimum(p, q).sum(-1)
            kl = (p * (torch.log(p.clamp_min(1e-12)) - F.log_softmax(
                student.float() / temperature, dim=-1))).sum(-1)
            overlap_sum += float(overlap.sum())
            tv_sum += float((1.0 - overlap).sum())
            kl_sum += float(kl.sum())
            top1_hits += int((p.argmax(-1) == q.argmax(-1)).sum())
            tokens += overlap.numel()
        kl_mean = kl_sum / tokens
        rows[temperature] = AlignmentRow(
            temperature=temperature,
            overlap=overlap_sum / tokens,
            tv=tv_sum / tokens,
            kl=kl_mean,
            top1=top1_hits / tokens,
            pinsker_lower_bound=pinsker_acceptance_lower_bound(kl_mean),
        ).to_dict()
        rows[temperature]["tokens"] = int(tokens)
    return rows


def theoretical_overlap_gap(overlap: float, measured_acceptance: float) -> dict:
    """Diagnose the gap between teacher-forced overlap and free-running acceptance."""
    return {
        "teacher_forced_overlap": overlap,
        "free_running_acceptance": measured_acceptance,
        "gap": overlap - measured_acceptance,
        "note": ("teacher forcing uses gold prefixes; free-running decoding uses "
                 "model-generated contexts, so exposure bias can open a gap"),
    }


def kl_to_overlap_bound(kl: float) -> float:
    """Alias kept explicit for report readability."""
    return 1.0 - math.sqrt(max(0.0, kl) / 2.0)
