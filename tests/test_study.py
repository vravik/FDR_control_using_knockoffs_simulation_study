"""Test the statistical components and reproducibility with small examples.

Run python -m unittest discover -s tests -v, or use make test.
"""

import json
from pathlib import Path
import unittest

import numpy as np

from src.dgps import DATA, make_rng, make_scenarios
from src.methods import NormalApproxMeanTest, TTest
from src.metrics import paired_difference
from src.simulation import run_replication


class StudyTests(unittest.TestCase):
    """Check DGP parameterization, mean tests, paired MCSEs, and task streams."""

    def test_dgp_parameterization(self):
        """Check generated means against independent theoretical values."""
        config = json.loads(Path("config.json").read_text())
        truths = {"Normal(0, 1)": 0.0, "Exponential(1)": 1.0,
                  "Lognormal(0, 1)": np.exp(0.5)}
        for scenario in make_scenarios(config["simulation"])[::2]:
            rng = make_rng(123, scenario.scenario_id, 0, purpose_id=DATA)
            sample = scenario.dgp.rvs(size=100_000, random_state=rng)
            self.assertEqual(sample.shape, (100_000,))
            self.assertAlmostEqual(scenario.dgp.mean(), truths[scenario.label])
            self.assertLess(abs(sample.mean() - truths[scenario.label]), 0.03)

    def test_method_correctness(self):
        """Check a centered null and invariance to a positive affine transform."""
        data = np.array([1., 2., 3., 4., 5.])
        # Mean=3, SE=1/sqrt(2): statistic=3*sqrt(2), with t reference df=4.
        expected = {1: 0.013235599563682681, 2: 0.00002209049699858536}
        for method in [TTest(), NormalApproxMeanTest()]:
            rng = np.random.default_rng(9)
            self.assertEqual(method.test(data, 3., rng)["p_value"], 1.)
            original = method.test(data, 0., rng)["p_value"]
            self.assertAlmostEqual(original, expected[method.method_id], places=12)
            transformed = method.test(2 * data + 7, 7., rng)["p_value"]
            self.assertAlmostEqual(original, transformed)

    def test_paired_mcse(self):
        """Check a known paired difference rather than an independence formula."""
        result = paired_difference([.1, .01, .02, .1], [.01, .1, .03, .1], .05)
        self.assertEqual(result["difference"], 0.)
        self.assertAlmostEqual(result["mcse"], np.sqrt(1 / 6))

    def test_repetition_replay_and_order(self):
        """Check that replay and method reordering preserve raw outputs."""
        config = json.loads(Path("config.json").read_text())
        scenario = make_scenarios(config["simulation"])[2]
        methods = [TTest(), NormalApproxMeanTest()]
        original = run_replication(scenario, methods, 7, 123)
        # Intervening tasks must not advance the streams for repetition 7.
        run_replication(scenario, methods, 2, 123)
        repeated = run_replication(scenario, methods[::-1], 7, 123)
        for a, b in zip(original, repeated[::-1]):
            self.assertEqual(a["p_value"], b["p_value"])
            self.assertEqual(a["method_rng_key"], b["method_rng_key"])


if __name__ == "__main__":
    unittest.main()
