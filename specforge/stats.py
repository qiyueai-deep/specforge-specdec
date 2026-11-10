"""Acceptance-process statistics for speculative decoding.

The verification loop of a speculative decoder is a discrete-time survival
process with right censoring:

* position j is *at risk* iff the first j-1 proposals were all accepted,
* a failure at position j occurs iff proposal j is the first rejection,
* proposals after the first rejection are never verified, they are right
  censored, not rejected.

Treating ``accepted / proposed`` as a per-position acceptance probability, or
treating ``P(A >= j) / #rounds`` as a conditional hazard, conflates the
survival function with the hazard and systematically biases every downstream
quantity. This module provides the correct estimators.

Pure standard library, so the statistical core can be unit-tested without
PyTorch and reused by every analysis script.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# Round-level bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    """One verification round of a speculative decoder.

    Mutable because the decoder fills in target/bonus counters as the round
    proceeds. Censoring fields are fixed as soon as verification ends.

    Attributes:
        proposed: number of draft tokens generated in this round (K_eff).
        accepted: number of proposals accepted (A).
        rejected: 1 if the round ended at the first rejection, else 0.
        verified: number of proposals actually scored by the target model.
                  Equals accepted + rejected. Everything else is censored.
        target_calls: number of target forward passes in this round.
        draft_steps: number of draft forward passes in this round.
        bonus: 1 if a bonus or replacement token was emitted.
        wall_seconds: optional wall-clock duration of the round.
    """

    proposed: int
    accepted: int
    rejected: int = 0
    verified: int = 0
    target_calls: int = 0
    draft_steps: int = 0
    bonus: int = 0
    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.proposed < 0 or self.accepted < 0 or self.rejected not in (0, 1):
            raise ValueError("invalid round record")
        if self.accepted > self.proposed:
            raise ValueError("accepted cannot exceed proposed")
        if self.verified < self.accepted + self.rejected:
            raise ValueError("verified must cover accepted and rejected proposals")

    @property
    def censored(self) -> int:
        """Draft tokens generated but never scored by the target."""
        return max(0, self.proposed - self.verified)

    def to_dict(self) -> dict:
        return {
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "verified": self.verified,
            "censored": self.censored,
            "target_calls": self.target_calls,
            "draft_steps": self.draft_steps,
            "bonus": self.bonus,
            "wall_seconds": self.wall_seconds,
        }


def from_legacy_rounds(rounds: Iterable[tuple[int, int, int]]) -> list[RoundRecord]:
    """Convert legacy ``(proposed, accepted, bonus)`` tuples to RoundRecords.

    The legacy tuple does not record whether a round ended in rejection, so a
    round is treated as rejected when ``accepted < proposed``. This is exactly
    the bookkeeping error the legacy instrumentation made, and is provided only
    so older JSON artifacts can still be re-analysed.
    """
    out = []
    for proposed, accepted, bonus in rounds:
        rejected = int(accepted < proposed)
        out.append(
            RoundRecord(
                proposed=proposed,
                accepted=accepted,
                rejected=rejected,
                verified=accepted + rejected,
                bonus=bonus,
                target_calls=2,
                draft_steps=proposed,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Acceptance-rate estimators
# ---------------------------------------------------------------------------


def verified_acceptance_rate(rounds: Sequence[RoundRecord]) -> float:
    """Unbiased per-position acceptance probability (the constant-hazard MLE).

    Denominator is the number of proposals the target actually scored, i.e.
    the exposed risk set. Censored tails are excluded.
    """
    verified = sum(r.verified for r in rounds)
    return sum(r.accepted for r in rounds) / max(1, verified)


def naive_acceptance_rate(rounds: Sequence[RoundRecord]) -> float:
    """Legacy ``accepted / proposed`` estimator, kept for audit only.

    It is downward biased with a bias that grows with K, because every
    rejected round contributes K-A-1 unverified tokens to the denominator.
    """
    proposed = sum(r.proposed for r in rounds)
    return sum(r.accepted for r in rounds) / max(1, proposed)


def acceptance_bias_report(rounds: Sequence[RoundRecord]) -> dict:
    proposed = sum(r.proposed for r in rounds)
    verified = sum(r.verified for r in rounds)
    accepted = sum(r.accepted for r in rounds)
    return {
        "rounds": len(rounds),
        "proposed": proposed,
        "verified": verified,
        "censored": proposed - verified,
        "accepted": accepted,
        "naive_rate": accepted / max(1, proposed),
        "verified_rate": accepted / max(1, verified),
        "censored_fraction": (proposed - verified) / max(1, proposed),
    }


# ---------------------------------------------------------------------------
# Wilson interval, bootstrap, incomplete gamma and chi-square tail
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[list[float]], float] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for an arbitrary statistic."""
    if not values:
        return (float("nan"), float("nan"))
    statistic = statistic or (lambda xs: sum(xs) / len(xs))
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(statistic(sample))
    draws.sort()
    lo = draws[min(n_boot - 1, int(alpha / 2 * n_boot))]
    hi = draws[min(n_boot - 1, max(0, int((1 - alpha / 2) * n_boot) - 1))]
    return (lo, hi)


def _gamma_series_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a,x), series expansion."""
    gln = math.lgamma(a)
    ap, total, delta = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gamma_cf_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a,x), continued fraction."""
    gln = math.lgamma(a)
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b if b != 0 else 1e300
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    """Survival function of a chi-square distribution, no SciPy needed."""
    if df <= 0 or x <= 0:
        return 1.0
    a, half_x = df / 2.0, x / 2.0
    if half_x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_series_p(a, half_x)))
    return max(0.0, min(1.0, _gamma_cf_q(a, half_x)))


# ---------------------------------------------------------------------------
# Discrete survival analysis of the acceptance process
# ---------------------------------------------------------------------------


@dataclass
class HazardEstimate:
    """Kaplan-Meier style estimate of the per-position acceptance hazard."""

    positions: list[int] = field(default_factory=list)
    risk_set: list[int] = field(default_factory=list)
    failures: list[int] = field(default_factory=list)
    hazard: list[float] = field(default_factory=list)
    hazard_ci: list[tuple[float, float]] = field(default_factory=list)
    survival: list[float] = field(default_factory=list)
    pooled_hazard: float = 0.0
    pooled_ci: tuple[float, float] = (0.0, 1.0)

    @property
    def max_position(self) -> int:
        return self.positions[-1] if self.positions else 0

    def survival_at(self, j: int) -> float:
        """S(j) = P(A >= j). Beyond the observed range, extrapolate with the
        pooled constant hazard (the fitted i.i.d. model)."""
        if j <= 0:
            return 1.0
        if j <= len(self.survival):
            return self.survival[j - 1]
        return self.survival[-1] * self.pooled_hazard ** (j - len(self.survival))

    def to_dict(self) -> dict:
        return {
            "positions": self.positions,
            "risk_set": self.risk_set,
            "failures": self.failures,
            "hazard": self.hazard,
            "hazard_ci": [list(c) for c in self.hazard_ci],
            "survival": self.survival,
            "pooled_hazard": self.pooled_hazard,
            "pooled_ci": list(self.pooled_ci),
        }


def risk_set_counts(rounds: Sequence[RoundRecord]) -> tuple[list[int], list[int]]:
    """Return per-position (n_j, d_j) for j = 1..max(K).

    n_j: rounds whose j-th proposal was actually verified (A >= j-1).
    d_j: rounds whose j-th proposal was the first rejection (A = j-1).
    Rounds that accept every proposal are right-censored after their last one.
    """
    k_max = max((r.proposed for r in rounds), default=0)
    n = [0] * (k_max + 1)
    d = [0] * (k_max + 1)
    for r in rounds:
        for j in range(1, r.proposed + 1):
            if r.accepted < j - 1:
                break
            n[j] += 1
            if r.rejected and r.accepted == j - 1:
                d[j] += 1
    return n[1:], d[1:]


def kaplan_meier(rounds: Sequence[RoundRecord]) -> HazardEstimate:
    """Estimate h_j = P(accept at j | j reached) with the correct risk set.

    This is the estimator that must be used when asking whether the i.i.d.
    geometric model holds. Under a constant hazard h, the estimated h_j is
    flat; the survival curve S(j) = product(h_i) = h**j is *not* flat, which
    is why plotting S(j) and calling it a hazard produces a spurious decay.
    """
    n, d = risk_set_counts(rounds)
    est = HazardEstimate()
    survival = 1.0
    for j, (nj, dj) in enumerate(zip(n, d), start=1):
        if nj == 0:
            break
        hj = 1.0 - dj / nj
        survival *= hj
        est.positions.append(j)
        est.risk_set.append(nj)
        est.failures.append(dj)
        est.hazard.append(hj)
        lo, hi = wilson_interval(nj - dj, nj)
        est.hazard_ci.append((lo, hi))
        est.survival.append(survival)
    total_n = sum(r.verified for r in rounds)
    total_a = sum(r.accepted for r in rounds)
    est.pooled_hazard = total_a / max(1, total_n)
    est.pooled_ci = wilson_interval(total_a, total_n)
    return est


def _bernoulli_loglik(n: int, d: int, h: float) -> float:
    if n <= 0:
        return 0.0
    eps = 1e-12
    h = min(1 - eps, max(eps, h))
    return d * math.log(1 - h) + (n - d) * math.log(h)


def lr_test_constant_hazard(
    rounds: Sequence[RoundRecord],
    min_risk: int = 5,
    n_sim: int = 0,
    seed: int = 0,
) -> dict:
    """Likelihood-ratio test: constant hazard H0 vs free per-position hazard H1.

    Returns the LR statistic, asymptotic chi-square p-value, and (optionally) a
    parametric-bootstrap p-value which is more reliable at the boundary where
    some d_j are 0 or n_j. A small p-value is evidence of genuine hazard
    heterogeneity, not of the censoring artefact.
    """
    n, d = risk_set_counts(rounds)
    used = [(nj, dj) for nj, dj in zip(n, d) if nj >= min_risk]
    if len(used) < 2:
        return {"positions": len(used), "lr": 0.0, "df": 0, "p_value": 1.0,
                "p_value_bootstrap": None, "note": "insufficient risk sets"}
    N, D = sum(nj for nj, _ in used), sum(dj for _, dj in used)
    h0 = 1.0 - D / N
    l0 = sum(_bernoulli_loglik(nj, dj, h0) for nj, dj in used)
    l1 = sum(_bernoulli_loglik(nj, dj, 1.0 - dj / nj) for nj, dj in used)
    lr = max(0.0, 2.0 * (l1 - l0))
    df = max(1, len(used) - 1)
    p = chi2_sf(lr, df)

    p_boot = None
    if n_sim > 0:
        rng = random.Random(seed)
        exceed = 0
        for _ in range(n_sim):
            sim = []
            for nj, _ in used:
                dj = sum(1 for _ in range(nj) if rng.random() > h0)
                sim.append((nj, dj))
            Ns, Ds = sum(a for a, _ in sim), sum(b for _, b in sim)
            hs = 1.0 - Ds / Ns
            ls0 = sum(_bernoulli_loglik(a, b, hs) for a, b in sim)
            ls1 = sum(_bernoulli_loglik(a, b, 1.0 - b / a) for a, b in sim)
            if 2.0 * (ls1 - ls0) >= lr:
                exceed += 1
        p_boot = (exceed + 1) / (n_sim + 1)
    return {
        "positions": len(used),
        "risk_exposures": N,
        "failures": D,
        "pooled_hazard": h0,
        "lr": lr,
        "df": df,
        "p_value": p,
        "p_value_bootstrap": p_boot,
        "heterogeneous": bool((p_boot if p_boot is not None else p) < 0.05),
    }


# ---------------------------------------------------------------------------
# Expected accepted tokens and distillation bounds
# ---------------------------------------------------------------------------


def expected_accepted_iid(a: float, k: int) -> float:
    """E[A] under a truncated geometric model: sum_{j=1}^{K} a**j."""
    if k <= 0:
        return 0.0
    if a >= 1.0 - 1e-12:
        return float(k)
    return a * (1.0 - a ** k) / (1.0 - a)


def expected_accepted_from_hazard(est: HazardEstimate, k: int) -> float:
    """E[A] = sum_{j=1}^{K} S(j), using the estimated survival curve."""
    return sum(est.survival_at(j) for j in range(1, k + 1))


def pinsker_acceptance_lower_bound(kl: float) -> float:
    """Stochastic-mode acceptance lower bound from Pinsker's inequality.

    E[accept] = sum min(p,q) = 1 - TV(p,q) >= 1 - sqrt(KL(p||q)/2).
    KL must be measured with natural logarithms at the *same* temperature as
    the decoding distribution.
    """
    return max(0.0, 1.0 - math.sqrt(max(0.0, kl) / 2.0))


# ---------------------------------------------------------------------------
# Two-sample distribution-equivalence tools
# ---------------------------------------------------------------------------


def total_variation(p: dict, q: dict) -> float:
    """Total variation distance between two empirical categorical distributions."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if sp <= 0 or sq <= 0:
        return float("nan")
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)


def permutation_tv_pvalue(
    sample_a: Sequence,
    sample_b: Sequence,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Permutation test on the TV distance between two categorical samples."""
    counts_a: dict = {}
    counts_b: dict = {}
    for x in sample_a:
        counts_a[x] = counts_a.get(x, 0) + 1
    for x in sample_b:
        counts_b[x] = counts_b.get(x, 0) + 1
    observed = total_variation(counts_a, counts_b)
    pool = list(sample_a) + list(sample_b)
    na = len(sample_a)
    rng = random.Random(seed)
    null = []
    for _ in range(n_perm):
        rng.shuffle(pool)
        ca: dict = {}
        cb: dict = {}
        for x in pool[:na]:
            ca[x] = ca.get(x, 0) + 1
        for x in pool[na:]:
            cb[x] = cb.get(x, 0) + 1
        null.append(total_variation(ca, cb))
    exceed = sum(1 for v in null if v >= observed - 1e-15)
    null.sort()
    critical = null[min(len(null) - 1, int(0.95 * len(null)))]
    return {
        "observed_tv": observed,
        "p_value": (exceed + 1) / (n_perm + 1),
        "null_tv_p95": critical,
        "n_a": na,
        "n_b": len(sample_b),
        "n_perm": n_perm,
    }


def detection_power(null_tv: Sequence[float], alternative_tv: Sequence[float], alpha: float = 0.05) -> float:
    """Empirical power: P(TV under the alternative exceeds the null critical value)."""
    if not null_tv or not alternative_tv:
        return float("nan")
    ordered = sorted(null_tv)
    critical = ordered[min(len(ordered) - 1, int((1 - alpha) * len(ordered)))]
    return sum(1 for v in alternative_tv if v > critical) / len(alternative_tv)
