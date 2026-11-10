from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import torch
from torch.nn import functional as F

from .model import TinyTransformerLM
from .stats import RoundRecord


@dataclass
class GenerationStats:
    generated_tokens: int = 0
    proposed_draft_tokens: int = 0
    # Only proposals actually scored by the target model belong in the risk
    # set. Everything after the first rejection is right-censored.
    verified_draft_tokens: int = 0
    censored_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    bonus_tokens: int = 0
    target_decode_calls: int = 0
    draft_decode_calls: int = 0
    target_processed_tokens: int = 0
    draft_processed_tokens: int = 0
    rollback_events: int = 0
    proposal_rounds: int = 0
    prefill_seconds: float = 0.0
    time_to_first_token_seconds: float = 0.0
    end_to_end_seconds: float = 0.0
    peak_target_cache_tokens: int = 0
    peak_draft_cache_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Per-position acceptance probability, estimated on the risk set.

        Denominator is ``verified_draft_tokens`` = accepted + rejected, not the
        number of drafted tokens. Using ``proposed_draft_tokens`` here treats
        right-censored proposals as rejections and biases alpha downward by an
        amount that grows with K.
        """
        if self.verified_draft_tokens:
            return self.accepted_draft_tokens / self.verified_draft_tokens
        return self.accepted_draft_tokens / max(1, self.proposed_draft_tokens)

    @property
    def naive_acceptance_rate(self) -> float:
        """Legacy accepted/proposed estimator, retained for audit only."""
        return self.accepted_draft_tokens / max(1, self.proposed_draft_tokens)

    @property
    def censored_fraction(self) -> float:
        return self.censored_draft_tokens / max(1, self.proposed_draft_tokens)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["acceptance_rate"] = self.acceptance_rate
        result["naive_acceptance_rate"] = self.naive_acceptance_rate
        result["censored_fraction"] = self.censored_fraction
        return result

    def mark_first_token(self, started: float) -> None:
        if self.time_to_first_token_seconds == 0.0:
            self.time_to_first_token_seconds = time.perf_counter() - started

    def observe_cache(self, target_tokens: int, draft_tokens: int = 0) -> None:
        self.peak_target_cache_tokens = max(self.peak_target_cache_tokens, target_tokens)
        self.peak_draft_cache_tokens = max(self.peak_draft_cache_tokens, draft_tokens)


def token_distribution(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits.float() / temperature
    if top_k is not None and 0 < top_k < scaled.shape[-1]:
        threshold = torch.topk(scaled, top_k).values[..., -1, None]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    return F.softmax(scaled, dim=-1)


def _sample(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


@torch.inference_mode()
def autoregressive_generate(
    model: TinyTransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    use_compile: bool = False,
) -> tuple[list[int], GenerationStats]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if mode not in {"greedy", "sample"}:
        raise ValueError("mode must be greedy or sample")
    if use_compile:
        return _autoregressive_generate_compiled(model, prompt_ids, max_new_tokens, mode, temperature, top_k, seed)
    started = time.perf_counter()
    cache = model.new_cache()
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    next_logits = model(prompt, cache=cache)[0, -1]
    generator = torch.Generator().manual_seed(seed)
    output = list(prompt_ids)
    stats = GenerationStats(prefill_seconds=time.perf_counter() - started)
    stats.observe_cache(cache.length)
    for _ in range(max_new_tokens):
        if mode == "greedy":
            token = int(next_logits.argmax())
        else:
            token = _sample(token_distribution(next_logits, temperature, top_k), generator)
        output.append(token)
        stats.mark_first_token(started)
        next_logits = model(torch.tensor([[token]], dtype=torch.long), cache=cache)[0, -1]
        stats.generated_tokens += 1
        stats.target_decode_calls += 1
        stats.target_processed_tokens += 1
        stats.observe_cache(cache.length)
    stats.end_to_end_seconds = time.perf_counter() - started
    return output, stats


def residual_distribution(target_prob: torch.Tensor, draft_prob: torch.Tensor) -> torch.Tensor:
    residual = torch.clamp(target_prob - draft_prob, min=0.0)
    total = residual.sum()
    return target_prob if float(total) <= 1e-12 else residual / total


@torch.inference_mode()
def speculative_generate(
    target: TinyTransformerLM,
    draft: TinyTransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    draft_k: int = 4,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    use_compile: bool = False,
    trace: list[RoundRecord] | None = None,
    k_policy: Callable[[int, int], int] | None = None,
) -> tuple[list[int], GenerationStats]:
    """Exact greedy speculation or distribution-preserving stochastic speculation.

    If ``trace`` is provided, one :class:`~specforge.stats.RoundRecord` per
    verification round is appended. It records the risk set explicitly, so all
    downstream statistics (hazard, survival, acceptance probability) use the
    *verified* proposals rather than the drafted ones. This is the single
    instrumented decode loop used by every analysis script.

    ``k_policy(remaining, generated)`` optionally chooses the candidate length
    per round, capped at ``draft_k``. This is how the cost-aware adaptive
    controller drives the decoder without duplicating the decode loop.
    The compiled path keeps a fixed K because its graph uses S=K padding.

    Greedy mode is bit-exact with target-only autoregressive decoding: a draft
    token is kept only when it equals the target argmax, otherwise the target
    argmax is emitted and both KV caches are rolled back.

    Stochastic mode implements rejection sampling.  Target and draft are always
    evaluated under the *same* temperature and top_k, so the acceptance
    probability ``min(1, p/q)`` and the residual ``normalize(max(p - q, 0))`` are
    defined over one identical distribution and the identity
    ``min(p,q) + (1 - sum min(p,q)) * normalize(max(p-q,0)) == p`` holds exactly
    (verified in correctness.py).

    Boundary worth stating in an interview: with ``top_k`` set, the identity
    holds on the *truncated* target distribution, not the untruncated one.  The
    sampler then reproduces "target sampled with top-k", which is a deliberate
    decoding policy, not an approximation of the raw target distribution.  Pass
    ``top_k=None`` when strict recovery of the untruncated distribution matters.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if draft_k < 1:
        raise ValueError("draft_k must be >= 1")
    if mode not in {"greedy", "sample"}:
        raise ValueError("mode must be greedy or sample")
    if use_compile:
        return _speculative_generate_compiled(
            target, draft, prompt_ids, max_new_tokens, draft_k, mode, temperature, top_k, seed, trace
        )
    started = time.perf_counter()
    target_cache, draft_cache = target.new_cache(), draft.new_cache()
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    target_next = target(prompt, cache=target_cache)[0, -1]
    draft_next = draft(prompt, cache=draft_cache)[0, -1]
    generator = torch.Generator().manual_seed(seed)
    output = list(prompt_ids)
    stats = GenerationStats(prefill_seconds=time.perf_counter() - started)
    stats.observe_cache(target_cache.length, draft_cache.length)

    while stats.generated_tokens < max_new_tokens:
        remaining = max_new_tokens - stats.generated_tokens
        if remaining == 1:
            probability = token_distribution(target_next, temperature, top_k)
            token = int(target_next.argmax()) if mode == "greedy" else _sample(probability, generator)
            output.append(token)
            stats.mark_first_token(started)
            target_next = target(torch.tensor([[token]], dtype=torch.long), cache=target_cache)[0, -1]
            draft_next = draft(torch.tensor([[token]], dtype=torch.long), cache=draft_cache)[0, -1]
            stats.generated_tokens += 1
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_cache.length, draft_cache.length)
            if trace is not None:
                trace.append(RoundRecord(proposed=0, accepted=0, target_calls=1,
                                        draft_steps=1, bonus=1))
            break

        current_k = (draft_k if k_policy is None
                     else max(1, min(draft_k, int(k_policy(remaining, stats.generated_tokens)))))
        proposal_count = min(current_k, remaining - 1)
        base_cache_len = target_cache.length
        proposals: list[int] = []
        draft_probabilities: list[torch.Tensor] = []
        for _ in range(proposal_count):
            probability = token_distribution(draft_next, temperature, top_k)
            token = int(draft_next.argmax()) if mode == "greedy" else _sample(probability, generator)
            proposals.append(token)
            draft_probabilities.append(probability)
            draft_next = draft(torch.tensor([[token]], dtype=torch.long), cache=draft_cache)[0, -1]
            stats.draft_decode_calls += 1
            stats.draft_processed_tokens += 1
        stats.proposed_draft_tokens += proposal_count
        stats.proposal_rounds += 1

        proposal_tensor = torch.tensor([proposals], dtype=torch.long)
        block_logits = target(proposal_tensor, cache=target_cache)[0]
        stats.target_decode_calls += 1
        stats.target_processed_tokens += proposal_count
        stats.observe_cache(target_cache.length, draft_cache.length)
        accepted = 0
        replacement: int | None = None

        for index, token in enumerate(proposals):
            verification_logits = target_next if index == 0 else block_logits[index - 1]
            if mode == "greedy":
                if token == int(verification_logits.argmax()):
                    accepted += 1
                    continue
                replacement = int(verification_logits.argmax())
                break
            target_probability = token_distribution(verification_logits, temperature, top_k)
            draft_probability = draft_probabilities[index]
            accept_probability = min(1.0, float(target_probability[token] / draft_probability[token].clamp_min(1e-12)))
            if float(torch.rand((), generator=generator)) <= accept_probability:
                accepted += 1
                continue
            replacement = _sample(residual_distribution(target_probability, draft_probability), generator)
            break

        stats.accepted_draft_tokens += accepted
        output.extend(proposals[:accepted])
        stats.generated_tokens += accepted
        if accepted:
            stats.mark_first_token(started)

        # Only the proposals up to (and including) the first rejection were
        # scored by the target. The rest are right-censored, not rejected.
        verified = accepted + (1 if replacement is not None else 0)
        stats.verified_draft_tokens += verified
        stats.censored_draft_tokens += proposal_count - verified
        record = RoundRecord(
            proposed=proposal_count,
            accepted=accepted,
            rejected=int(replacement is not None),
            verified=verified,
            target_calls=1,
            draft_steps=proposal_count,
        )
        if trace is not None:
            trace.append(record)

        if replacement is not None:
            stats.rejected_draft_tokens += 1
            stats.rollback_events += 1
            record.target_calls += 1
            record.draft_steps += 1
            target_cache.truncate(base_cache_len + accepted)
            draft_cache.truncate(base_cache_len + accepted)
            output.append(replacement)
            stats.generated_tokens += 1
            stats.mark_first_token(started)
            token_tensor = torch.tensor([[replacement]], dtype=torch.long)
            target_next = target(token_tensor, cache=target_cache)[0, -1]
            draft_next = draft(token_tensor, cache=draft_cache)[0, -1]
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_cache.length, draft_cache.length)
            continue

        target_next = block_logits[-1]
        if stats.generated_tokens < max_new_tokens:
            bonus_probability = token_distribution(target_next, temperature, top_k)
            bonus = int(target_next.argmax()) if mode == "greedy" else _sample(bonus_probability, generator)
            output.append(bonus)
            stats.generated_tokens += 1
            stats.mark_first_token(started)
            stats.bonus_tokens += 1
            record.target_calls += 1
            record.draft_steps += 1
            record.bonus = 1
            bonus_tensor = torch.tensor([[bonus]], dtype=torch.long)
            target_next = target(bonus_tensor, cache=target_cache)[0, -1]
            draft_next = draft(bonus_tensor, cache=draft_cache)[0, -1]
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_cache.length, draft_cache.length)

    stats.end_to_end_seconds = time.perf_counter() - started
    return output, stats


# ---------------------------------------------------------------------------
# torch.compile 加速路径（use_compile=True）
#
# 与 eager 版本逐位等价的算法，但模型前向换用 forward_functional 的
# torch.compile 编译版：KV Cache 以张量列表传入/返回（不再变异 KVCache
# 对象），回滚直接切片。torch.compile 的 CPU 后端需要 MSVC；在无编译器
# 环境下调用会抛 RuntimeError，由调用方决定是否回退 eager。
# ---------------------------------------------------------------------------


def _empty_past(model: TinyTransformerLM) -> tuple[torch.Tensor, torch.Tensor]:
    """Empty stacked KV tensors [n_layers, 1, H, 0, D] for the functional decode path."""
    head_dim = model.config.d_model // model.config.n_heads
    keys = torch.empty(model.config.n_layers, 1, model.config.n_heads, 0, head_dim)
    values = torch.empty(model.config.n_layers, 1, model.config.n_heads, 0, head_dim)
    return keys, values


def _compiled_forward(model: TinyTransformerLM):
    """Compile the functional forward once per model instance and cache it.

    ``dynamic=True`` marks every dimension dynamic so the cache-length dim T
    grows freely without recompiling; the sequence dim S is held to a small
    fixed set of values by the callers (prefill=prompt len, decode=1,
    block verification=padded to draft_k), so at most a handful of graphs are
    compiled per model and reused for the whole run.
    """
    compiled = getattr(model, "_specforge_compiled_forward", None)
    if compiled is None:
        compiled = torch.compile(model.forward_functional, dynamic=True)
        model._specforge_compiled_forward = compiled
    return compiled


@torch.inference_mode()
def _autoregressive_generate_compiled(
    model: TinyTransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    mode: str,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> tuple[list[int], GenerationStats]:
    forward = _compiled_forward(model)
    started = time.perf_counter()
    past_keys, past_values = _empty_past(model)
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    next_logits, past_keys, past_values = forward(prompt, past_keys, past_values)
    next_logits = next_logits[0, -1]
    generator = torch.Generator().manual_seed(seed)
    output = list(prompt_ids)
    stats = GenerationStats(prefill_seconds=time.perf_counter() - started)
    stats.observe_cache(past_keys.shape[3])
    for _ in range(max_new_tokens):
        if mode == "greedy":
            token = int(next_logits.argmax())
        else:
            token = _sample(token_distribution(next_logits, temperature, top_k), generator)
        output.append(token)
        stats.mark_first_token(started)
        token_tensor = torch.tensor([[token]], dtype=torch.long)
        next_logits, past_keys, past_values = forward(token_tensor, past_keys, past_values)
        next_logits = next_logits[0, -1]
        stats.generated_tokens += 1
        stats.target_decode_calls += 1
        stats.target_processed_tokens += 1
        stats.observe_cache(past_keys.shape[3])
    stats.end_to_end_seconds = time.perf_counter() - started
    return output, stats


@torch.inference_mode()
def _speculative_generate_compiled(
    target: TinyTransformerLM,
    draft: TinyTransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    draft_k: int,
    mode: str,
    temperature: float,
    top_k: int | None,
    seed: int,
    trace: list[RoundRecord] | None = None,
) -> tuple[list[int], GenerationStats]:
    target_forward = _compiled_forward(target)
    draft_forward = _compiled_forward(draft)
    started = time.perf_counter()
    target_keys, target_values = _empty_past(target)
    draft_keys, draft_values = _empty_past(draft)
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    target_next, target_keys, target_values = target_forward(prompt, target_keys, target_values)
    target_next = target_next[0, -1]
    draft_next, draft_keys, draft_values = draft_forward(prompt, draft_keys, draft_values)
    draft_next = draft_next[0, -1]
    generator = torch.Generator().manual_seed(seed)
    output = list(prompt_ids)
    stats = GenerationStats(prefill_seconds=time.perf_counter() - started)
    stats.observe_cache(target_keys.shape[3], draft_keys.shape[3])

    while stats.generated_tokens < max_new_tokens:
        remaining = max_new_tokens - stats.generated_tokens
        if remaining == 1:
            probability = token_distribution(target_next, temperature, top_k)
            token = int(target_next.argmax()) if mode == "greedy" else _sample(probability, generator)
            output.append(token)
            stats.mark_first_token(started)
            token_tensor = torch.tensor([[token]], dtype=torch.long)
            target_next, target_keys, target_values = target_forward(token_tensor, target_keys, target_values)
            target_next = target_next[0, -1]
            draft_next, draft_keys, draft_values = draft_forward(token_tensor, draft_keys, draft_values)
            draft_next = draft_next[0, -1]
            stats.generated_tokens += 1
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_keys.shape[3], draft_keys.shape[3])
            if trace is not None:
                trace.append(RoundRecord(proposed=0, accepted=0, target_calls=1,
                                        draft_steps=1, bonus=1))
            break

        current_k = draft_k
        proposal_count = min(current_k, remaining - 1)
        base_cache_len = target_keys.shape[3]
        proposals: list[int] = []
        draft_probabilities: list[torch.Tensor] = []
        for _ in range(proposal_count):
            probability = token_distribution(draft_next, temperature, top_k)
            token = int(draft_next.argmax()) if mode == "greedy" else _sample(probability, generator)
            proposals.append(token)
            draft_probabilities.append(probability)
            token_tensor = torch.tensor([[token]], dtype=torch.long)
            draft_next, draft_keys, draft_values = draft_forward(token_tensor, draft_keys, draft_values)
            draft_next = draft_next[0, -1]
            stats.draft_decode_calls += 1
            stats.draft_processed_tokens += 1
        stats.proposed_draft_tokens += proposal_count
        stats.proposal_rounds += 1

        proposal_tensor = torch.tensor([proposals], dtype=torch.long)
        # 固定 S=K 的块验证：不足 K 个提议时用最后一个 draft token 填充到 K，
        # 使编译版块验证调用始终保持同一输入序列长（避免 torch.compile 按长度重编译）。
        # 填充位置对应的 logits 不参与接受判定，其产生的缓存垃圾在下方无条件回滚中清除。
        pad = current_k - proposal_count
        if pad > 0:
            proposal_tensor = torch.cat(
                [proposal_tensor, torch.full((1, pad), proposals[-1], dtype=torch.long)], dim=1
            )
        block_logits, target_keys, target_values = target_forward(proposal_tensor, target_keys, target_values)
        block_logits = block_logits[0]
        stats.target_decode_calls += 1
        stats.target_processed_tokens += proposal_count
        stats.observe_cache(target_keys.shape[3], draft_keys.shape[3])
        accepted = 0
        replacement: int | None = None

        for index, token in enumerate(proposals):
            verification_logits = target_next if index == 0 else block_logits[index - 1]
            if mode == "greedy":
                if token == int(verification_logits.argmax()):
                    accepted += 1
                    continue
                replacement = int(verification_logits.argmax())
                break
            target_probability = token_distribution(verification_logits, temperature, top_k)
            draft_probability = draft_probabilities[index]
            accept_probability = min(1.0, float(target_probability[token] / draft_probability[token].clamp_min(1e-12)))
            if float(torch.rand((), generator=generator)) <= accept_probability:
                accepted += 1
                continue
            replacement = _sample(residual_distribution(target_probability, draft_probability), generator)
            break

        stats.accepted_draft_tokens += accepted
        output.extend(proposals[:accepted])
        stats.generated_tokens += accepted
        if accepted:
            stats.mark_first_token(started)

        # Only the proposals up to (and including) the first rejection were
        # scored by the target. The rest are right-censored, not rejected.
        verified = accepted + (1 if replacement is not None else 0)
        stats.verified_draft_tokens += verified
        stats.censored_draft_tokens += proposal_count - verified
        record = RoundRecord(
            proposed=proposal_count,
            accepted=accepted,
            rejected=int(replacement is not None),
            verified=verified,
            target_calls=1,
            draft_steps=proposal_count,
        )
        if trace is not None:
            trace.append(record)

        # 编译版按固定 K 填充块验证，目标缓存可能多出 K - proposal_count 个填充垃圾；
        # 无论接受与否都回滚到真实长度 base + accepted（全接受时此切片是必要清理）。
        new_cache_len = base_cache_len + accepted
        target_keys = target_keys[:, :, :, :new_cache_len]
        target_values = target_values[:, :, :, :new_cache_len]

        if replacement is not None:
            stats.rejected_draft_tokens += 1
            stats.rollback_events += 1
            record.target_calls += 1
            record.draft_steps += 1
            draft_keys = draft_keys[:, :, :, :new_cache_len]
            draft_values = draft_values[:, :, :, :new_cache_len]
            output.append(replacement)
            stats.generated_tokens += 1
            stats.mark_first_token(started)
            token_tensor = torch.tensor([[replacement]], dtype=torch.long)
            target_next, target_keys, target_values = target_forward(token_tensor, target_keys, target_values)
            target_next = target_next[0, -1]
            draft_next, draft_keys, draft_values = draft_forward(token_tensor, draft_keys, draft_values)
            draft_next = draft_next[0, -1]
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_keys.shape[3], draft_keys.shape[3])
            continue

        target_next = block_logits[proposal_count - 1]
        if stats.generated_tokens < max_new_tokens:
            bonus_probability = token_distribution(target_next, temperature, top_k)
            bonus = int(target_next.argmax()) if mode == "greedy" else _sample(bonus_probability, generator)
            output.append(bonus)
            stats.generated_tokens += 1
            stats.mark_first_token(started)
            stats.bonus_tokens += 1
            record.target_calls += 1
            record.draft_steps += 1
            record.bonus = 1
            token_tensor = torch.tensor([[bonus]], dtype=torch.long)
            target_next, target_keys, target_values = target_forward(token_tensor, target_keys, target_values)
            target_next = target_next[0, -1]
            draft_next, draft_keys, draft_values = draft_forward(token_tensor, draft_keys, draft_values)
            draft_next = draft_next[0, -1]
            stats.target_decode_calls += 1
            stats.draft_decode_calls += 1
            stats.target_processed_tokens += 1
            stats.draft_processed_tokens += 1
            stats.observe_cache(target_keys.shape[3], draft_keys.shape[3])

    stats.end_to_end_seconds = time.perf_counter() - started
    return output, stats
