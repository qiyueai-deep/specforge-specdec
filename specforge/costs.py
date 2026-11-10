"""Cost-aware candidate-length selection for speculative decoding.

The v1 project used a confidence threshold on the draft softmax as an early
exit. That heuristic had two problems: it was motivated by a spurious hazard
decay (see analyze_acceptance_theory.py), and on low-entropy data the threshold
never fired. Here the decision is driven by a measured wall-clock cost model.

Model
-----
One verification round with K proposals costs C(K) seconds (measured, not
assumed) and produces E[A(K)] + 1 tokens (accepted proposals plus one bonus or
replacement token). Under a constant per-position acceptance hazard h,

    E[A(K)] = sum_{j=1}^{K} S(j) = h (1 - h**K) / (1 - h)

so the expected seconds per generated token is

    tau(K) = C(K) / (E[A(K)] + 1),      K* = argmin_K tau(K).

Honesty about the optimality claim: if h and C are stationary, the optimisation
problem is identical in every round, so the best adaptive policy collapses to
the best fixed K. A dynamic policy can only *strictly* win when h or C is
heterogeneous across prompts or non-stationary within a generation. The
controller below estimates h online and is evaluated against the best fixed K
on a held-out set, with confidence intervals.

All costs come from the production decoder itself. There is no duplicate
decode loop in this module.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from .decoding import speculative_generate
from .stats import RoundRecord, expected_accepted_iid, wilson_interval


@dataclass
class CostModel:
    """Measured per-round wall-clock cost, indexed by candidate length."""

    round_seconds_by_k: dict[int, float] = field(default_factory=dict)
    intercept: float = 0.0
    slope: float = 0.0
    r_squared: float = 0.0
    micro: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def round_seconds(self, k: int) -> float:
        """Measured cost, with linear interpolation/extrapolation for unseen K."""
        table = self.round_seconds_by_k
        if not table:
            raise ValueError("empty cost model")
        if k in table:
            return table[k]
        ks = sorted(table)
        if k <= ks[0]:
            return max(1e-9, table[ks[0]] + self.slope * (k - ks[0]))
        if k >= ks[-1]:
            return max(1e-9, table[ks[-1]] + self.slope * (k - ks[-1]))
        lo = max(x for x in ks if x < k)
        hi = min(x for x in ks if x > k)
        w = (k - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    def single_token_cost(self) -> float:
        """Cost of the terminal round where only one token remains."""
        micro = self.micro or {}
        t = micro.get("target_single_token_seconds")
        d = micro.get("draft_single_token_seconds")
        if t is not None and d is not None:
            return t + d
        return max(1e-9, self.round_seconds(1) * 0.7)

    def to_dict(self) -> dict:
        return {
            "round_seconds_by_k": self.round_seconds_by_k,
            "intercept": self.intercept,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "micro": self.micro,
            "metadata": self.metadata,
        }


def _linear_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares a + b*x, plus R^2."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0 or n < 2:
        return (my, 0.0, 0.0)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    sse = sum((y - (a + b * x)) ** 2 for x, y in points)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return (a, b, r2)


def _time_once(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def microbenchmark_components(target, draft, prompt_ids: list[int], repeats: int = 40) -> dict:
    """Measure the elementary forward costs that make up one round.

    A fresh cache is prefilled before each timed call so past length is held
    fixed. This decomposition is explanatory, not used for the policy.
    """
    torch.set_num_threads(1)
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    one = torch.tensor([[prompt_ids[-1]]], dtype=torch.long)

    def block_cost(model, tokens: torch.Tensor, k: int) -> float:
        samples = []
        for _ in range(repeats):
            cache = model.new_cache()
            with torch.inference_mode():
                model(prompt, cache=cache)
                samples.append(_time_once(lambda: model(tokens, cache=cache)))
        return statistics.median(samples)

    target_single = block_cost(target, one, 1)
    draft_single = block_cost(draft, one, 1)
    verify_by_k = {}
    for k in (1, 4, 8, 16, 28, 32):
        block = torch.tensor([[prompt_ids[-1]] * k], dtype=torch.long)
        verify_by_k[k] = block_cost(target, block, k)
    a, b, r2 = _linear_fit([(k, v) for k, v in verify_by_k.items()])
    return {
        "target_single_token_seconds": target_single,
        "draft_single_token_seconds": draft_single,
        "target_block_seconds_by_k": verify_by_k,
        "verify_intercept_seconds": a,
        "verify_slope_seconds_per_token": b,
        "verify_linear_r_squared": r2,
        "torch_threads": torch.get_num_threads(),
    }


def profile_round_costs(
    target,
    draft,
    prompts: Sequence[list[int]],
    k_values: Sequence[int],
    new_tokens: int = 64,
    repeats: int = 3,
    warmups: int = 1,
    mode: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    micro_repeats: int = 20,
) -> tuple[CostModel, dict]:
    """Measure C(K) = total wall time / number of verification rounds.

    Uses ``speculative_generate`` directly, so the measured cost includes all
    Python, softmax, tensor construction and cache-truncation overhead. Raw
    per-prompt samples are returned so the caller can bootstrap intervals.
    """
    torch.set_num_threads(1)
    raw: dict[int, list[float]] = {}
    per_round: dict[int, float] = {}
    for k in k_values:
        samples: list[float] = []
        per_round_samples: list[float] = []
        for prompt in prompts:
            for _ in range(warmups):
                speculative_generate(target, draft, prompt, new_tokens, draft_k=k,
                                     mode=mode, temperature=temperature, top_k=top_k)
            for _ in range(repeats):
                start = time.perf_counter()
                _, stats = speculative_generate(
                    target, draft, prompt, new_tokens, draft_k=k,
                    mode=mode, temperature=temperature, top_k=top_k,
                )
                seconds = time.perf_counter() - start
                samples.append(seconds)
                # Normalise by *this* generation's round count. Using a single
                # reference prompt's round count for every prompt mixes up
                # different K_eff regimes and distorts C(K).
                per_round_samples.append(seconds / max(1, stats.proposal_rounds))
        raw[k] = samples
        per_round[k] = statistics.median(per_round_samples)

    a, b, r2 = _linear_fit([(k, per_round[k]) for k in sorted(per_round)])
    micro = microbenchmark_components(target, draft, prompts[0], repeats=micro_repeats)
    model = CostModel(
        round_seconds_by_k=per_round,
        intercept=a,
        slope=b,
        r_squared=r2,
        micro=micro,
        metadata={
            "new_tokens": new_tokens,
            "repeats": repeats,
            "warmups": warmups,
            "prompt_count": len(prompts),
            "mode": mode,
            "temperature": temperature,
            "top_k": top_k,
            "device": "cpu",
            "torch_threads": 1,
        },
    )
    return model, raw


# ---------------------------------------------------------------------------
# Optimal stopping / policy
# ---------------------------------------------------------------------------


def expected_tokens_per_round(hazard: float, k: int) -> float:
    return expected_accepted_iid(hazard, k) + 1.0


def seconds_per_token(model: CostModel, hazard: float, k: int) -> float:
    """tau(K) = C(K) / (E[A(K)] + 1)."""
    k = max(1, int(k))
    return model.round_seconds(k) / max(1e-9, expected_tokens_per_round(hazard, k))


def optimal_fixed_k(
    model: CostModel,
    hazard: float,
    k_max: int,
    remaining: int | None = None,
) -> tuple[int, list[dict]]:
    """K* = argmin tau(K). Fixed K is a member of the policy space, so no
    adaptive policy can beat this when hazard and cost are stationary."""
    upper = k_max if remaining is None else min(k_max, max(1, remaining - 1))
    table = [
        {"k": k, "round_seconds": model.round_seconds(k),
         "expected_tokens": expected_tokens_per_round(hazard, k),
         "seconds_per_token": seconds_per_token(model, hazard, k)}
        for k in range(1, upper + 1)
    ]
    best = min(table, key=lambda row: row["seconds_per_token"])
    return best["k"], table


class DPKPlanner:
    """Finite-horizon Bellman-optimal candidate-length policy.

    State ``n`` is the number of tokens still to generate, the action is
    ``K in 1..min(k_max, n-1)``, and under a constant hazard h the number of
    accepted proposals is truncated geometric, so one round emits A+1 tokens:

        V(n) = min_K [ C(K) + E_h[ V(n - A - 1) ] ],   V(0) = 0,
        V(1) = single_token_cost().

    Minimising tau(K) round by round is only the infinite-horizon greedy
    solution. The DP also handles the end of generation, where a myopic small
    K creates extra verification rounds. Within the model, and when h and C
    are stationary, this policy is globally optimal, hence at least as good as
    any fixed K. Tables are cached per binned hazard so the per-round decision
    is O(1).
    """

    def __init__(
        self,
        model: CostModel,
        k_max: int,
        max_tokens: int,
        hazard_grid_step: float = 0.01,
        h_min: float = 0.40,
        h_max: float = 0.999,
    ) -> None:
        self.model = model
        self.k_max = k_max
        self.max_tokens = max_tokens
        self.hazard_grid_step = hazard_grid_step
        self.h_min = h_min
        self.h_max = h_max
        self._cache: dict[int, dict] = {}

    def _solve(self, h: float) -> dict:
        value = [0.0] * (self.max_tokens + 1)
        policy = [1] * (self.max_tokens + 1)
        value[1] = self.model.single_token_cost()
        for n in range(2, self.max_tokens + 1):
            upper = min(self.k_max, n - 1)
            best_value, best_k = float("inf"), 1
            for k in range(1, upper + 1):
                cost = self.model.round_seconds(k)
                expectation = 0.0
                probability = 1.0 - h
                power = 1.0
                for a in range(k):
                    expectation += probability * value[max(0, n - a - 1)]
                    power *= h
                    probability = (1.0 - h) * power
                expectation += (h ** k) * value[max(0, n - k - 1)]
                total = cost + expectation
                if total < best_value:
                    best_value, best_k = total, k
            value[n] = best_value
            policy[n] = best_k
        return {"value": value, "policy": policy}

    def table(self, hazard: float) -> dict:
        h = min(self.h_max, max(self.h_min, float(hazard)))
        key = int(round(h / self.hazard_grid_step))
        hq = min(self.h_max, max(self.h_min, key * self.hazard_grid_step))
        if key not in self._cache:
            self._cache[key] = self._solve(hq)
        return self._cache[key]

    def choose(self, hazard: float, remaining: int) -> int:
        n = min(self.max_tokens, max(1, int(remaining)))
        return self.table(hazard)["policy"][n]

    def expected_seconds(self, hazard: float, remaining: int) -> float:
        n = min(self.max_tokens, max(1, int(remaining)))
        return self.table(hazard)["value"][n]


class AdaptiveKController:
    """Online hazard estimation + per-round cost-optimal candidate length.

    The prior is a Beta(1,1) smoothed estimate. A Wilson lower bound can be
    used instead of the point estimate when a conservative policy is wanted.
    ``make_policy(trace)`` returns the callable consumed by
    ``speculative_generate(..., k_policy=...)``, so the controller reads the
    very same RoundRecords that the decoder emits.
    """

    def __init__(
        self,
        model: CostModel,
        k_max: int,
        prior_hazard: float = 0.90,
        prior_strength: float = 12.0,
        conservative: bool = False,
        fallback_k: int | None = None,
        planner: DPKPlanner | None = None,
    ) -> None:
        self.model = model
        self.k_max = k_max
        self.prior_hazard = prior_hazard
        self.prior_strength = prior_strength
        self.conservative = conservative
        self.fallback_k = fallback_k
        self.planner = planner
        self.accepted = 0
        self.verified = 0
        self._seen = 0
        self.chosen: list[int] = []

    @property
    def hazard(self) -> float:
        successes = self.accepted + self.prior_hazard * self.prior_strength
        trials = self.verified + self.prior_strength
        h = successes / trials
        if not self.conservative:
            return h
        lo, _ = wilson_interval(self.accepted, self.verified) if self.verified else (h, h)
        return max(0.5, min(h, lo if self.verified >= 10 else h))

    def observe(self, record: RoundRecord) -> None:
        if record.proposed <= 0:
            return
        self.accepted += record.accepted
        self.verified += record.verified

    def choose(self, remaining: int) -> int:
        fallback = self.fallback_k if self.fallback_k is not None else self.k_max
        if self.verified == 0:
            k = fallback
        elif self.planner is not None:
            k = self.planner.choose(self.hazard, remaining)
        else:
            k, _ = optimal_fixed_k(self.model, self.hazard, self.k_max, remaining)
        k = max(1, min(self.k_max, max(1, remaining - 1), k))
        self.chosen.append(k)
        return k

    def make_policy(self, trace: list[RoundRecord]):
        def policy(remaining: int, generated: int) -> int:
            if len(trace) > self._seen:
                for record in trace[self._seen:]:
                    self.observe(record)
                self._seen = len(trace)
            return self.choose(remaining)
        return policy

    def to_dict(self) -> dict:
        return {
            "k_max": self.k_max,
            "prior_hazard": self.prior_hazard,
            "prior_strength": self.prior_strength,
            "conservative": self.conservative,
            "fallback_k": self.fallback_k,
            "planner": "bellman_dp" if self.planner is not None else "greedy_tau",
            "accepted": self.accepted,
            "verified": self.verified,
            "final_hazard": self.hazard,
            "chosen_k": self.chosen,
            "mean_chosen_k": statistics.fmean(self.chosen) if self.chosen else None,
        }


def theoretical_speedup(model: CostModel, hazard: float, k: int, baseline_token_seconds: float) -> float:
    """Predicted speedup over target-only autoregressive decoding."""
    return baseline_token_seconds / max(1e-12, seconds_per_token(model, hazard, k))


def gini(values: Sequence[float]) -> float:
    """Heterogeneity of per-prompt hazards, used to explain when adaptivity can win."""
    xs = sorted(float(v) for v in values if not math.isnan(v))
    if not xs or statistics.fmean(xs) == 0:
        return 0.0
    n = len(xs)
    total = sum(xs)
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * total)
