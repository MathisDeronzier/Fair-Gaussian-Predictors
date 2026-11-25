"""Public empirical transport API checks, including documented defaults."""

import unittest

import torch

from Fair_GPs.utils import (
    empirical_quantile, w2_squared_1d, wasserstein2_barycenter_1d,
)


class FairRegressionTests(unittest.TestCase):
    def test_empirical_quantile_at_jump_boundaries(self):
        sample = torch.tensor([-2., 1., 5.])
        levels = torch.tensor([0., 1 / 3, 0.34, 2 / 3, 1.])
        torch.testing.assert_close(empirical_quantile(sample, levels),
                                   torch.tensor([-2., -2., 1., 1., 5.]))

    def test_unequal_sample_sizes_have_exact_transport_cost(self):
        x, y = torch.tensor([0., 2.]), torch.tensor([1.])
        self.assertEqual(float(w2_squared_1d(x, y)), 1.)
        self.assertEqual(float(w2_squared_1d(y, x)), 1.)

    def test_documented_missing_weight_and_point_mass_barycenter(self):
        x, y = torch.tensor([0.], requires_grad=True), torch.tensor([2.], requires_grad=True)
        actual = wasserstein2_barycenter_1d(x, y, p0=0.25)
        explicit = wasserstein2_barycenter_1d(x, y, p0=0.25, p1=0.75)
        for a, b in zip(actual, explicit):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(actual.TX, torch.tensor([1.5]))
        torch.testing.assert_close(actual.TY, torch.tensor([1.5]))
        self.assertEqual(float(actual.value), 0.75)
        actual.value.backward()
        torch.testing.assert_close(x.grad, torch.tensor([-0.75]))
        torch.testing.assert_close(y.grad, torch.tensor([0.75]))

    def test_automatic_sample_proportions_and_invalid_weights(self):
        x, y = torch.tensor([0., 2.]), torch.tensor([1.])
        actual = wasserstein2_barycenter_1d(x, y, p0=None)
        explicit = wasserstein2_barycenter_1d(x, y, p0=2 / 3, p1=1 / 3)
        for a, b in zip(actual, explicit):
            torch.testing.assert_close(a, b)
        for p0, p1 in [(-0.1, 1.1), (0.2, 0.2), (float("nan"), 0.5)]:
            with self.subTest(p0=p0, p1=p1), self.assertRaises(ValueError):
                wasserstein2_barycenter_1d(x, y, p0=p0, p1=p1)


if __name__ == "__main__":
    unittest.main()
