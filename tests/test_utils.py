"""Shared distribution diagnostics and utility import behavior."""

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import ks_2samp
import torch

from Fair_GPs.utils import KS, ks_from_empirical_cdfs, true_cdf


class UtilityTests(unittest.TestCase):
    def test_shared_ks_matches_scipy_with_ties_and_unequal_groups(self):
        values = torch.tensor([0., 2., 0., 1., 3., 3., 1.])
        sensitive = torch.tensor([0., 1., 0., 1., 1., 1., 0.])
        expected = float(ks_2samp(values[sensitive == 0], values[sensitive == 1]).statistic)
        for order in (torch.arange(7), torch.tensor([6, 1, 4, 2, 5, 0, 3])):
            for column in (False, True):
                x, s = values[order], sensitive[order]
                if column:
                    x, s = x[:, None], s[:, None]
                actual = KS(x, s)
                self.assertIsInstance(actual, float)
                self.assertEqual(actual, expected)
                self.assertEqual(ks_from_empirical_cdfs(x, s), expected)

    def test_cdf_audit_does_not_call_the_scipy_implementation(self):
        values = torch.tensor([0., 1., 1., 2.])
        sensitive = torch.tensor([0., 0., 1., 1.])
        with patch("Fair_GPs.utils.stats.ks_2samp", side_effect=AssertionError("SciPy used")):
            self.assertEqual(ks_from_empirical_cdfs(values, sensitive), 0.5)

    def test_empirical_cdf_counts_ties_on_an_ordered_grid(self):
        sample = torch.tensor([0., 0., 1., 3.])
        grid = torch.tensor([-1., 0., 0.5, 1., 2., 3.])
        np.testing.assert_array_equal(true_cdf(grid, sample), [0., 0.5, 0.5, 0.75, 0.75, 1.])

    def test_ks_rejects_invalid_or_missing_groups(self):
        for values, sensitive in (
            (torch.tensor([1., 2.]), torch.tensor([0., 0.])),
            (torch.tensor([1., 2.]), torch.tensor([0., 2.])),
            (torch.tensor([1., float("nan")]), torch.tensor([0., 1.])),
            (torch.tensor([1., 2.]), torch.tensor([0.])),
        ):
            for function in (KS, ks_from_empirical_cdfs):
                with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                    function(values, sensitive)

    def test_import_preserves_torch_defaults(self):
        code = """
import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')
before = (torch.get_default_dtype(), torch.empty(0).device)
import Fair_GPs.utils
after = (torch.get_default_dtype(), torch.empty(0).device)
assert after == before, (before, after)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ks_keeps_cpu_inputs_on_cpu_with_another_default_device(self):
        values = torch.tensor([0., 1., 1., 2.], device="cpu")
        sensitive = torch.tensor([0., 0., 1., 1.], device="cpu")
        previous_device = torch.empty(0).device
        try:
            torch.set_default_device("meta")
            self.assertEqual(KS(values, sensitive), 0.5)
            self.assertEqual(ks_from_empirical_cdfs(values, sensitive), 0.5)
            self.assertEqual(KS(values.numpy(), sensitive.numpy()), 0.5)
        finally:
            torch.set_default_device(previous_device)


if __name__ == "__main__":
    unittest.main()
