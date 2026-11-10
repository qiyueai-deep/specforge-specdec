"""Unit tests for the acceptance-process statistics (pure standard library).

These tests encode the central methodological correction of the project:
under a genuinely i.i.d. constant hazard, the legacy estimator produces a
spurious decaying "per-position acceptance" curve, while the risk-set
estimator stays flat. No PyTorch is required to run this file.
"""
from __future__ import annotations

import math
import random
import unittest

from specforge.stats import (
    RoundRecord,
    acceptance_bias_report,
    bootstrap_ci,
    chi2_sf,
    detection_power,
    expected_accepted_from_hazard,
    expected_accepted_iid,
    kaplan_meier,
    lr_test_constant_hazard,
    naive_acceptance_rate,
    permutation_tv_pvalue,
    pinsker_acceptance_lower_bound,
    risk_set_counts,
    total_variation,
    verified_acceptance_rate,
    wilson_interval,
)


def simulate_iid_rounds(a: float, k: int, n_rounds: int, seed: int = 0) -> list[RoundRecord]:
    """Generate rounds from a *true* constant hazard (the i.i.d. null)."""
    rng = random.Random(seed)
    rounds = []
    for _ in range(n_rounds):
        accepted = 0
        rejected = 0
        while accepted < k:
            if rng.random() < a:
                accepted += 1
            else:
                rejected = 1
                break
        rounds.append(
            RoundRecord(
                proposed=k,
                accepted=accepted,
                rejected=rejected,
                verified=accepted + rejected,
                target_calls=2,
                draft_steps=k + 1,
                bonus=1,
            )
        )
    return rounds


class TestRiskSetAccounting(unittest.TestCase):
    def test_censored_tokens_are_not_rejections(self):
        r = RoundRecord(proposed=10, accepted=3, rejected=1, verified=4)
        self.assertEqual(r.censored, 6)
        self.assertEqual(verified_acceptance_rate([r]), 0.75)
        self.assertEqual(naive_acceptance_rate([r]), 0.30)

    def test_risk_set_counts(self):
        # One round accepts 2 then rejects at position 3, one round accepts all 4.
        rounds = [
            RoundRecord(proposed=4, accepted=2, rejected=1, verified=3),
            RoundRecord(proposed=4, accepted=4, rejected=0, verified=4),
        ]
        n, d = risk_set_counts(rounds)
        self.assertEqual(n, [2, 2, 2, 1])
        self.assertEqual(d, [0, 0, 1, 0])

    def test_bias_report(self):
        rounds = simulate_iid_rounds(0.97, 28, 4000, seed=3)
        rep = acceptance_bias_report(rounds)
        self.assertGreater(rep["censored"], 0)
        self.assertGreater(rep["verified_rate"], rep["naive_rate"])
        self.assertAlmostEqual(rep["verified_rate"], 0.97, delta=0.02)


class TestConstantHazardNull(unittest.TestCase):
    def test_hazard_is_flat_under_iid(self):
        rounds = simulate_iid_rounds(0.97, 28, 20000, seed=11)
        est = kaplan_meier(rounds)
        self.assertAlmostEqual(est.pooled_hazard, 0.97, delta=0.01)
        for j in (0, 5, 10, 20, 27):
            self.assertAlmostEqual(est.hazard[j], 0.97, delta=0.02)
        # The survival function decays even though the hazard is constant.
        self.assertLess(est.survival[24], est.survival[0])

    def test_legacy_estimator_fakes_a_decay(self):
        """Reproduce the artefact: S(j) plotted as if it were a hazard."""
        rounds = simulate_iid_rounds(0.986, 28, 20000, seed=12)
        n, d = risk_set_counts(rounds)
        n_total = len(rounds)
        legacy = [sum(1 for r in rounds if r.accepted >= j) / n_total for j in range(1, 6)]
        self.assertGreater(legacy[0], 0.97)
        self.assertLess(legacy[-1], legacy[0])
        est = kaplan_meier(rounds)
        self.assertAlmostEqual(est.hazard[0], est.hazard[4], delta=0.01)

    def test_lr_does_not_reject_iid(self):
        rounds = simulate_iid_rounds(0.95, 12, 3000, seed=13)
        result = lr_test_constant_hazard(rounds, min_risk=20, n_sim=300, seed=13)
        self.assertFalse(result["heterogeneous"])
        self.assertGreater(result["p_value_bootstrap"], 0.05)

    def test_lr_detects_real_heterogeneity(self):
        rng = random.Random(21)
        rounds = []
        for _ in range(4000):
            accepted, rejected = 0, 0
            while accepted < 12:
                h = 0.99 if accepted < 3 else 0.80
                if rng.random() < h:
                    accepted += 1
                else:
                    rejected = 1
                    break
            rounds.append(RoundRecord(proposed=12, accepted=accepted, rejected=rejected,
                                      verified=accepted + rejected))
        result = lr_test_constant_hazard(rounds, min_risk=20, n_sim=200, seed=21)
        self.assertTrue(result["heterogeneous"])

    def test_naive_bias_grows_with_k(self):
        biases = []
        for k in (4, 12, 28):
            rounds = simulate_iid_rounds(0.97, k, 20000, seed=17)
            biases.append(verified_acceptance_rate(rounds) - naive_acceptance_rate(rounds))
        self.assertLess(biases[0], biases[1])
        self.assertLess(biases[1], biases[2])


class TestAnalyticHelpers(unittest.TestCase):
    def test_chi2_sf_known_values(self):
        self.assertAlmostEqual(chi2_sf(3.841458820694124, 1), 0.05, delta=1e-6)
        self.assertAlmostEqual(chi2_sf(5.991464547107979, 2), 0.05, delta=1e-5)
        self.assertAlmostEqual(chi2_sf(0.0, 3), 1.0)

    def test_wilson_interval_covers(self):
        lo, hi = wilson_interval(970, 1000)
        self.assertLess(lo, 0.97)
        self.assertGreater(hi, 0.97)

    def test_expected_accepted_matches_monte_carlo(self):
        a, k = 0.95, 20
        predicted = expected_accepted_iid(a, k)
        rounds = simulate_iid_rounds(a, k, 40000, seed=31)
        measured = sum(r.accepted for r in rounds) / len(rounds)
        self.assertAlmostEqual(predicted, measured, delta=0.1)
        est = kaplan_meier(rounds)
        self.assertAlmostEqual(expected_accepted_from_hazard(est, k), measured, delta=0.15)

    def test_pinsker_bound(self):
        self.assertAlmostEqual(pinsker_acceptance_lower_bound(0.0128), 1 - math.sqrt(0.0064), delta=1e-9)
        self.assertGreater(pinsker_acceptance_lower_bound(0.01), pinsker_acceptance_lower_bound(0.10))

    def test_bootstrap_ci(self):
        lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=800, seed=5)
        self.assertLess(lo, 3.0)
        self.assertGreater(hi, 3.0)


class TestDistributionEquivalence(unittest.TestCase):
    def test_total_variation(self):
        self.assertAlmostEqual(total_variation({"a": 1}, {"a": 1}), 0.0)
        self.assertAlmostEqual(total_variation({"a": 1}, {"b": 1}), 1.0)

    def test_permutation_identical_distributions(self):
        rng = random.Random(41)
        a = [rng.randrange(5) for _ in range(500)]
        b = [rng.randrange(5) for _ in range(500)]
        result = permutation_tv_pvalue(a, b, n_perm=400, seed=41)
        self.assertGreater(result["p_value"], 0.05)

    def test_permutation_detects_shift(self):
        rng = random.Random(42)
        a = [0 if rng.random() < 0.9 else 1 for _ in range(500)]
        b = [1 if rng.random() < 0.9 else 0 for _ in range(500)]
        result = permutation_tv_pvalue(a, b, n_perm=400, seed=42)
        self.assertLess(result["p_value"], 0.01)

    def test_detection_power(self):
        self.assertAlmostEqual(detection_power([0.01] * 100, [0.5] * 100), 1.0)
        self.assertAlmostEqual(detection_power([0.01] * 100, [0.001] * 100), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
