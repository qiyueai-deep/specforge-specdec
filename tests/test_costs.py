"""Tests for the cost model and the Bellman-optimal K planner."""
from __future__ import annotations

import random
import statistics
import unittest

from specforge.costs import (
    CostModel,
    DPKPlanner,
    expected_tokens_per_round,
    optimal_fixed_k,
    seconds_per_token,
)


def fake_cost_model() -> CostModel:
    table = {k: (5.0 + 0.8 * k) / 1000.0 for k in range(1, 33)}
    return CostModel(
        round_seconds_by_k=table,
        intercept=5.0 / 1000.0,
        slope=0.8 / 1000.0,
        micro={
            "target_single_token_seconds": 0.002,
            "draft_single_token_seconds": 0.0005,
        },
    )


def simulate_policy(hazard, chooser, model, max_tokens=64, n_sim=3000, seed=7) -> float:
    rng = random.Random(seed)
    totals = []
    for _ in range(n_sim):
        left, cost = max_tokens, 0.0
        while left > 0:
            if left == 1:
                cost += model.single_token_cost()
                left = 0
                break
            k = max(1, min(chooser(left), left - 1))
            cost += model.round_seconds(k)
            accepted = 0
            for _ in range(k):
                if rng.random() < hazard:
                    accepted += 1
                else:
                    break
            left -= accepted + 1
        totals.append(cost)
    return statistics.fmean(totals)


class TestCostModel(unittest.TestCase):
    def test_round_seconds_lookup_and_extrapolation(self):
        m = fake_cost_model()
        self.assertAlmostEqual(m.round_seconds(8), (5 + 6.4) / 1000)
        self.assertGreater(m.round_seconds(40), m.round_seconds(32))
        self.assertAlmostEqual(m.single_token_cost(), 0.0025)

    def test_expected_tokens_monotone(self):
        self.assertLess(expected_tokens_per_round(0.9, 4), expected_tokens_per_round(0.9, 8))
        self.assertAlmostEqual(expected_tokens_per_round(1.0, 7), 8.0)

    def test_optimal_fixed_k(self):
        m = fake_cost_model()
        k, table = optimal_fixed_k(m, 0.95, 32)
        self.assertTrue(1 <= k <= 32)
        self.assertAlmostEqual(table[k - 1]["seconds_per_token"],
                               min(r["seconds_per_token"] for r in table))

    def test_seconds_per_token(self):
        m = fake_cost_model()
        self.assertAlmostEqual(seconds_per_token(m, 0.9, 10),
                               m.round_seconds(10) / expected_tokens_per_round(0.9, 10))


class TestBellmanPlanner(unittest.TestCase):
    def test_policy_range_and_value_monotonicity(self):
        m = fake_cost_model()
        planner = DPKPlanner(m, k_max=32, max_tokens=48)
        table = planner.table(0.9)
        for n in range(1, 49):
            self.assertTrue(1 <= table["policy"][n] <= min(32, max(1, n - 1)))
        self.assertTrue(all(table["value"][i] <= table["value"][i + 1] + 1e-12
                            for i in range(0, 48)))

    def test_dp_is_optimal_against_every_fixed_k(self):
        """Core claim: fixed K is a subset of the Bellman policy space."""
        m = fake_cost_model()
        h = 0.88
        planner = DPKPlanner(m, k_max=32, max_tokens=64)
        dp_value = planner.expected_seconds(h, 64)
        for k in (4, 8, 16, 28):
            fixed = simulate_policy(h, lambda _n, k=k: k, m, n_sim=2500, seed=100 + k)
            self.assertLessEqual(dp_value, fixed + 0.002,
                                 f"DP value {dp_value} should not exceed fixed K={k} cost {fixed}")

    def test_dp_simulation_beats_long_fixed_k_on_low_hazard(self):
        """With a low hazard, a long fixed K wastes draft steps; DP avoids it."""
        m = fake_cost_model()
        h = 0.70
        planner = DPKPlanner(m, k_max=32, max_tokens=64)
        dp = simulate_policy(h, lambda n: planner.choose(h, n), m, n_sim=2500, seed=31)
        long_k = simulate_policy(h, lambda _n: 28, m, n_sim=2500, seed=31)
        self.assertLess(dp, long_k)

    def test_hazard_binning_is_stable(self):
        m = fake_cost_model()
        planner = DPKPlanner(m, k_max=16, max_tokens=32)
        a = planner.table(0.901)["policy"]
        b = planner.table(0.902)["policy"]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
