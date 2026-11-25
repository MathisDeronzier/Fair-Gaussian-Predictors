"""Regression tests for singular MAP prediction and decomposition reuse."""

import unittest
from unittest.mock import patch

import numpy as np
import gpytorch
import torch

from Fair_GPs.Fair_GPs_classe import (
    GP, apply_symmetric_pinv, predict_map_path,
    symmetric_pinv_factors, symmetric_pseudoinverse,
)


class PseudoinverseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.options = dict(lengthscale=0.3, sigma=1.7, device=torch.device("cpu"))
        self.x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        self.z = torch.tensor([1., 3., 2., 6.])

    def test_moore_penrose_identities_and_independent_svd(self):
        generator = torch.Generator().manual_seed(91)
        q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator))
        gram = (q * torch.tensor([4., 2., 0.7, 0., 0., 0.])) @ q.T
        inverse = symmetric_pseudoinverse(gram)
        expected = np.linalg.pinv(gram.numpy(), rcond=6 * np.finfo(np.float64).eps)
        np.testing.assert_allclose(inverse.numpy(), expected, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(gram @ inverse @ gram, gram, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(inverse @ gram @ inverse, inverse, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(gram @ inverse, (gram @ inverse).T, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(inverse @ gram, (inverse @ gram).T, atol=1e-13, rtol=1e-13)

    def test_singular_rho_endpoints_project_incompatible_values(self):
        for rho, expected in [(1., [2., 2., 4., 4.]), (-1., [-1., 1., -2., 2.])]:
            with self.subTest(rho=rho):
                gp = GP(self.x, self.z, Map_solution=self.z, rho=rho, **self.options)
                actual = gp.MAP_pred(self.x.numpy(), jitter=0)
                torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-12, rtol=1e-12)
                self.assertEqual(gp.map_pinv_info["rank"], 2)
                self.assertGreater(gp.map_pinv_info["discarded_component_norm"], 0)
                self.assertTrue(torch.isfinite(gp.Map_coefficients).all())
                # No exact interpolant exists for these incompatible duplicate inputs.
                self.assertGreater(float((actual - self.z).abs().max()), 0.5)

    def test_compatible_values_interpolate_at_both_endpoints(self):
        for rho in (-1., 1.):
            values = torch.tensor([1., rho, 2., 2. * rho])
            gp = GP(self.x, values, Map_solution=values, rho=rho, **self.options)
            torch.testing.assert_close(gp.MAP_pred(self.x), values, atol=1e-12, rtol=1e-12)
            self.assertLess(gp.map_pinv_info["discarded_component_norm"], 1e-12)

    def test_nonsingular_predictions_match_exact_solve(self):
        gp = GP(self.x, self.z, Map_solution=self.z, rho=0.3, **self.options)
        test = torch.tensor([[0.2, 0.], [0.6, 1.]])
        for jitter in (0., 1e-5):
            gram = gp.get_gram(self.x)
            expected = gp.get_gram(test, self.x) @ torch.linalg.solve(
                gram + jitter * torch.eye(len(gram)), self.z
            )
            torch.testing.assert_close(gp.MAP_pred(test, jitter=jitter), expected, atol=1e-12, rtol=1e-12)
            self.assertEqual(gp.map_pinv_info["rank"], 4)
            self.assertEqual(gp.map_pinv_info["jitter"], jitter)

    def test_decomposition_is_reused_and_invalidated_with_kernel(self):
        gp = GP(self.x, self.z, Map_solution=self.z, rho=1., **self.options)
        with patch("Fair_GPs.Fair_GPs_classe.torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            gp.MAP_pred(self.x)
            gp.MAP_pred(self.x[:2])
            gp.MAP_solution = self.z + 1
            changed = gp.MAP_pred(self.x)
            gp.get_pinv_gram()
            self.assertEqual(eigh.call_count, 1)
            torch.testing.assert_close(changed, torch.tensor([3., 3., 5., 5.]))
            gp.MAP_solution.add_(2)
            torch.testing.assert_close(gp.MAP_pred(self.x), torch.tensor([5., 5., 7., 7.]))
            self.assertEqual(eigh.call_count, 1)
            # GPyTorch setters may mutate .data without changing tensor versions.
            gp.set_hyperparameters(lengthscale=0.6)
            gp.MAP_pred(self.x)
            self.assertEqual(eigh.call_count, 2)
            gp.MAP_pred(self.x, jitter=1e-6)
            gp.MAP_pred(self.x, jitter=1e-6, rtol=1e-10)
            self.assertEqual(eigh.call_count, 4)
            gp.Update_data(self.x, self.z)
            gp.MAP_pred(self.x)
            self.assertEqual(eigh.call_count, 5)

    def test_cached_coefficients_follow_solution_and_rho(self):
        gp = GP(self.x, self.z, Map_solution=self.z.clone(), rho=0.3, **self.options)
        gp.MAP_pred(self.x)
        gp.MAP_solution.data.add_(0.5)
        torch.testing.assert_close(gp.MAP_pred(self.x), self.z + 0.5)
        gp.set_hyperparameters(rho=1.)
        torch.testing.assert_close(gp.MAP_pred(self.x), torch.tensor([2.5, 2.5, 4.5, 4.5]))
        gp.set_hyperparameters(rho=-1.)
        torch.testing.assert_close(gp.MAP_pred(self.x), torch.tensor([-1., 1., -2., 2.]))

    def test_multiple_rhs_and_supplied_coefficients(self):
        gp = GP(self.x, self.z, Map_solution=self.z, rho=1., **self.options)
        values = torch.stack((self.z, self.z + 2, -self.z), dim=1)
        coefficients = gp.compute_map_coefficients(values)
        torch.testing.assert_close(gp.get_gram(self.x) @ coefficients,
                                   torch.tensor([[2., 4., -2.], [2., 4., -2.],
                                                 [4., 6., -4.], [4., 6., -4.]]))
        restored = GP(self.x, self.z, Map_solution=self.z,
                      Map_coefficients=coefficients[:, 0], rho=1., **self.options)
        with patch("Fair_GPs.Fair_GPs_classe.torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            torch.testing.assert_close(restored.MAP_pred(self.x), torch.tensor([2., 2., 4., 4.]))
            self.assertEqual(eigh.call_count, 0)
            restored.MAP_solution = self.z + 1
            restored.MAP_pred(self.x)
            self.assertEqual(eigh.call_count, 1)

    def test_cutoff_is_explicit_and_zero_matrix_is_supported(self):
        gram = torch.diag(torch.tensor([2., 1e-10, 0.]))
        _, _, automatic = symmetric_pinv_factors(gram)
        self.assertEqual(automatic["rank"], 2)
        inverse = symmetric_pseudoinverse(gram, rtol=1e-8)
        torch.testing.assert_close(inverse, torch.diag(torch.tensor([0.5, 0., 0.])))
        _, _, absolute = symmetric_pinv_factors(gram, atol=1e-8)
        self.assertEqual(absolute["effective_rtol"], 0.)
        self.assertEqual(absolute["rank"], 1)
        torch.testing.assert_close(symmetric_pseudoinverse(torch.zeros(3, 3)), torch.zeros(3, 3))

    def test_vector_application_has_no_broadcasting(self):
        gram = torch.diag(torch.tensor([2., 0., 4.]))
        vectors, reciprocals, _ = symmetric_pinv_factors(gram)
        values = torch.tensor([2., 3., 4.])
        actual = apply_symmetric_pinv(vectors, reciprocals, values)
        torch.testing.assert_close(actual, torch.tensor([1., 0., 1.]))
        self.assertEqual(tuple(actual.shape), (3,))

    def test_diagnostics_follow_cached_predictions(self):
        gp = GP(self.x, self.z, Map_solution=self.z, **self.options)
        gp.MAP_pred(self.x, rtol=1e-8)
        info = dict(gp.map_pinv_info)
        gp.get_pinv_gram(self.x[:2])
        self.assertEqual(gp.map_pinv_info["size"], 2)
        gp.MAP_pred(self.x, rtol=1e-8)
        self.assertEqual(gp.map_pinv_info, info)

    def test_batched_helper_matches_public_predictions_and_reuses_factors(self):
        gp = GP(self.x, self.z, Map_solution=self.z, rho=1., **self.options)
        solutions = [self.z, self.z + 2, -self.z]
        with patch("Fair_GPs.Fair_GPs_classe.torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            observed, cf0, cf1 = predict_map_path(gp, self.x, solutions)
            for index, values in enumerate(solutions):
                gp.MAP_solution = values
                torch.testing.assert_close(observed[index], gp.MAP_pred(self.x))
            torch.testing.assert_close(cf0, cf1)
            predict_map_path(gp, self.x[:2], solutions)
            self.assertEqual(eigh.call_count, 1)

    def test_custom_kernel_nontensor_changes_invalidate_factors(self):
        class CustomKernel(gpytorch.kernels.RBFKernel):
            gain = 1.

            def forward(self, x1, x2, **params):
                return self.gain * super().forward(x1, x2, **params)

        kernel = CustomKernel()
        gp = GP(self.x, self.z, Map_solution=self.z, kernel_x=kernel, rho=0.3, **self.options)
        with patch("Fair_GPs.Fair_GPs_classe.torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            before = gp.MAP_pred(self.x)
            coefficients = gp.Map_coefficients.clone()
            kernel.gain = 2.
            torch.testing.assert_close(gp.MAP_pred(self.x), before)
            torch.testing.assert_close(gp.Map_coefficients, coefficients / 2)
            self.assertEqual(eigh.call_count, 2)

    def test_invalid_settings_and_values_are_rejected(self):
        gp = GP(self.x, self.z, Map_solution=self.z, **self.options)
        for options in ({"jitter": -1}, {"rtol": -1}, {"atol": float("nan")}, {"rtol": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                gp.MAP_pred(self.x, **options)
        for values in (torch.ones(3), torch.ones(4, 0), torch.full((4,), float("nan"))):
            with self.assertRaises(ValueError):
                gp.compute_map_coefficients(values)
        for matrix in (torch.ones(2, 3), torch.zeros(0, 0), torch.ones(2, 2, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                symmetric_pseudoinverse(matrix)


if __name__ == "__main__":
    unittest.main()
