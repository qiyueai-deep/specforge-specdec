"""Statistical verification of distribution-preserving stochastic decoding.

The project's v1 correctness check only verified an algebraic identity on
randomly generated probability vectors:

    min(p,q) + (1 - sum min(p,q)) * normalize(max(p-q,0)) == p

That proves a one-step formula, not that the *decoder* samples from the target
distribution. This module upgrades the claim to an auditable statistical one:

* draw N sequences from target-only sampling and N from speculative sampling,
* estimate the total-variation distance of the token and bigram distributions,
* test H0 "both samples come from one common distribution" with a parametric
  bootstrap under the pooled empirical law,
* bootstrap a confidence interval for the observed TV,
* report the detection power of the test at a given effect size delta.

A non-significant result is only meaningful together with power: without it,
"we failed to reject" says nothing. All estimators here are two-sample,
distribution-free with respect to the model's functional form.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch

from .decoding import autoregressive_generate, speculative_generate
from .stats import detection_power, total_variation


def sample_sequences(
    target,
    draft,
    prompt: list[int],
    n_samples: int,
    n_tokens: int,
    draft_k: int = 8,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed_start: int = 10000,
) -> tuple[list[list[int]], list[list[int]]]:
    """Draw continuation sequences from both decoders under identical settings."""
    target_only, speculative = [], []
    for i in range(n_samples):
        seed = seed_start + i
        out, _ = autoregressive_generate(
            target, prompt, n_tokens, mode="sample",
            temperature=temperature, top_k=top_k, seed=seed,
        )
        target_only.append(out[len(prompt):])
        out, _ = speculative_generate(
            target, draft, prompt, n_tokens, draft_k=draft_k, mode="sample",
            temperature=temperature, top_k=top_k, seed=seed,
        )
        speculative.append(out[len(prompt):])
    return target_only, speculative


def ngram_counter(sequences: Sequence[Sequence[int]], n: int = 1) -> Counter:
    counter: Counter = Counter()
    for seq in sequences:
        if n == 1:
            counter.update(seq)
        else:
            counter.update(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    return counter


def _count_matrix(sequences: Sequence[Sequence[int]], vocab_size: int, n: int = 1) -> np.ndarray:
    """One row per sequence, one column per token id (unigram only)."""
    matrix = np.zeros((len(sequences), vocab_size), dtype=np.float64)
    for i, seq in enumerate(sequences):
        for token in seq:
            if 0 <= int(token) < vocab_size:
                matrix[i, int(token)] += 1.0
    return matrix


def _tv_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(0.5 * np.abs(a / sa - b / sb).sum())


def parametric_bootstrap_tv_pvalue(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_sim: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """H0: both samples are draws from the same pooled categorical law.

    Under H0 the sufficient statistics are multinomial(n_a, p_pool) and
    multinomial(n_b, p_pool). Simulating them gives the null distribution of TV
    without assuming any parametric form for p itself.
    """
    na, nb = int(counts_a.sum()), int(counts_b.sum())
    pooled = (counts_a + counts_b) / max(1, na + nb)
    observed = _tv_from_arrays(counts_a, counts_b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_sim)
    for i in range(n_sim):
        ca = rng.multinomial(na, pooled).astype(np.float64)
        cb = rng.multinomial(nb, pooled).astype(np.float64)
        null[i] = 0.5 * np.abs(ca / na - cb / nb).sum()
    critical = float(np.quantile(null, 1.0 - alpha))
    p_value = (1 + int((null >= observed - 1e-15).sum())) / (n_sim + 1)
    return {
        "observed_tv": observed,
        "null_tv_mean": float(null.mean()),
        "null_tv_critical": critical,
        "p_value": p_value,
        "reject_h0": bool(p_value < alpha),
        "n_sim": n_sim,
        "alpha": alpha,
    }


def tv_bootstrap_ci(
    sequences_a: Sequence[Sequence[int]],
    sequences_b: Sequence[Sequence[int]],
    vocab_size: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI for TV, resampling whole sequences (not tokens)."""
    ma = _count_matrix(sequences_a, vocab_size)
    mb = _count_matrix(sequences_b, vocab_size)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(ma), len(ma))
        ib = rng.integers(0, len(mb), len(mb))
        draws.append(_tv_from_arrays(ma[ia].sum(0), mb[ib].sum(0)))
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


def power_at_effect(
    pooled: np.ndarray,
    effect: float,
    n_a: int,
    n_b: int,
    null_tv: np.ndarray,
    n_sim: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Empirical power against an alternative with exactly TV = effect.

    Mass ``effect`` is moved from the most likely category to another one, so
    TV(p, q) = effect by construction.
    """
    rng = np.random.default_rng(seed)
    p = pooled.copy()
    q = pooled.copy()
    source = int(np.argmax(p))
    sink = int(np.argmin(p))
    delta = min(effect, p[source] - 1e-12)
    if delta <= 0:
        return float("nan")
    q[source] -= delta
    q[sink] += delta
    alternative = np.empty(n_sim)
    for i in range(n_sim):
        ca = rng.multinomial(n_a, p).astype(np.float64)
        cb = rng.multinomial(n_b, q).astype(np.float64)
        alternative[i] = 0.5 * np.abs(ca / n_a - cb / n_b).sum()
    return detection_power(null_tv.tolist(), alternative.tolist(), alpha)


@dataclass
class EquivalenceResult:
    n_sequences: int
    tokens_per_sequence: int
    draft_k: int
    temperature: float
    observed_tv_unigram: float
    tv_ci_unigram: list[float]
    p_value: float
    null_tv_critical: float
    observed_tv_bigram: float
    reject_h0: bool
    power_curve: list[dict]
    exact_greedy: bool
    spec_seconds: float
    baseline_seconds: float
    stochastic_speedup: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_equivalence_suite(
    target,
    draft,
    prompt: list[int],
    vocab_size: int,
    n_samples: int = 400,
    n_tokens: int = 24,
    draft_k: int = 8,
    temperature: float = 1.0,
    top_k: int | None = None,
    effects=(0.01, 0.02, 0.05, 0.10),
    seed: int = 0,
) -> EquivalenceResult:
    """Full equivalence audit plus a stochastic-mode wall-clock measurement."""
    import time

    torch.set_num_threads(1)
    a, b = sample_sequences(target, draft, prompt, n_samples, n_tokens,
                            draft_k, temperature, top_k, seed_start=10000 + seed)

    ca = ngram_counter(a, 1)
    cb = ngram_counter(b, 1)
    counts_a = np.zeros(vocab_size)
    counts_b = np.zeros(vocab_size)
    for key, value in ca.items():
        if 0 <= int(key) < vocab_size:
            counts_a[int(key)] = value
    for key, value in cb.items():
        if 0 <= int(key) < vocab_size:
            counts_b[int(key)] = value
    boot = parametric_bootstrap_tv_pvalue(counts_a, counts_b, n_sim=2000, seed=seed)
    ci = tv_bootstrap_ci(a, b, vocab_size, n_boot=1000, seed=seed + 1)
    bigram_tv = total_variation(dict(ngram_counter(a, 2)), dict(ngram_counter(b, 2)))

    pooled = (counts_a + counts_b) / max(1, counts_a.sum() + counts_b.sum())
    rng = np.random.default_rng(seed + 2)
    null_tv = np.array([
        0.5 * np.abs(rng.multinomial(int(counts_a.sum()), pooled) / counts_a.sum()
                     - rng.multinomial(int(counts_b.sum()), pooled) / counts_b.sum()).sum()
        for _ in range(1000)
    ])
    curve = [{"effect": e,
              "power": power_at_effect(pooled, e, int(counts_a.sum()), int(counts_b.sum()),
                                       null_tv, n_sim=200, seed=seed + int(e * 10000))}
             for e in effects]

    # Greedy exactness and a paired stochastic-mode wall-clock measurement.
    greedy_base, _ = autoregressive_generate(target, prompt, n_tokens, mode="greedy")
    greedy_spec, _ = speculative_generate(target, draft, prompt, n_tokens,
                                          draft_k=draft_k, mode="greedy")
    exact = bool(greedy_base == greedy_spec)

    def timed(fn, repeats=5):
        for _ in range(1):
            fn()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
        return float(np.median(samples))

    baseline_seconds = timed(lambda: autoregressive_generate(
        target, prompt, n_tokens, mode="sample", temperature=temperature,
        top_k=top_k, seed=4242))
    spec_seconds = timed(lambda: speculative_generate(
        target, draft, prompt, n_tokens, draft_k=draft_k, mode="sample",
        temperature=temperature, top_k=top_k, seed=4242))

    return EquivalenceResult(
        n_sequences=n_samples,
        tokens_per_sequence=n_tokens,
        draft_k=draft_k,
        temperature=temperature,
        observed_tv_unigram=boot["observed_tv"],
        tv_ci_unigram=list(ci),
        p_value=boot["p_value"],
        null_tv_critical=boot["null_tv_critical"],
        observed_tv_bigram=float(bigram_tv),
        reject_h0=boot["reject_h0"],
        power_curve=curve,
        exact_greedy=exact,
        spec_seconds=spec_seconds,
        baseline_seconds=baseline_seconds,
        stochastic_speedup=baseline_seconds / max(1e-9, spec_seconds),
    )
