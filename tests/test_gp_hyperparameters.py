import unittest

import numpy as np
import torch

from Fair_GPs.Fair_GPs_classe import GP
from Fair_GPs.Data_Processing import normalize_train_test


class GPFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(21)
        spatial = torch.rand(16, generator=generator, dtype=torch.float64)
        codes = torch.arange(16) % 3
        groups = torch.arange(16) % 2
        self.x = torch.stack((spatial, codes, groups), dim=1)
        self.y = torch.sin(6 * spatial) + 0.2 * codes - 0.4 * groups
        self.options = dict(rho=-0.3, categorical_dims=[1], categorical_corr=0.4, device=torch.device("cpu"))

    def test_categorical_covariance_depends_on_matches(self):
        x = torch.tensor([[0., 10., 0.], [1., 10., 0.], [0., 99., 1.], [1., 99., 1.]])
        gp = GP(x, torch.arange(4.), rho=-0.3, sigma=2., categorical_dims=[0, 1],
                categorical_corr=0.4, device=torch.device("cpu"))
        gram = gp.get_gram(x)
        expected = torch.tensor([
            [2., 0.8, -0.24, -0.096],
            [0.8, 2., -0.096, -0.24],
            [-0.24, -0.096, 2., 0.8],
            [-0.096, -0.24, 0.8, 2.],
        ], dtype=gram.dtype)
        torch.testing.assert_close(gram, expected)
        self.assertEqual(len(gp.hyperparameters.lengthscale), 0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(gram).min()), 0.)
        recoded = x.clone()
        recoded[:, 0] = recoded[:, 0] * 25 - 17
        recoded[:, 1] = recoded[:, 1] * -6
        torch.testing.assert_close(gp.get_gram(recoded), expected)
        torch.testing.assert_close(gp.model.covar_module(x, diag=True), gram.diag())

    def test_scaling_keeps_unseen_category_codes(self):
        test_x = self.x[:3].clone()
        test_x[:, 1] = 99
        train, target, test, _, info = normalize_train_test(
            self.x, self.y, test_x, self.y[:3], categorical_dims=[1], center_targets=True
        )
        torch.testing.assert_close(train[:, 1:], self.x[:, 1:])
        torch.testing.assert_close(test[:, 1:], test_x[:, 1:])
        self.assertAlmostEqual(float(target.mean()), 0.)
        self.assertEqual(info["continuous_dims"], (0,))

    def test_joint_design_retains_winning_full_state(self):
        for design, threshold in [("sobol", 100), ("lhs", 5)]:
            with self.subTest(design=design):
                def build():
                    torch.manual_seed(25)
                    return GP(self.x, self.y, sparse_threshold=threshold, inducing_m=6,
                              varnoise=0.02, lengthscale=0.6, fix_lengthscale=True, **self.options)

                gp = build()
                hp = gp.Optimise_hyperparameters(
                    fix_varnoise=True, fix_rho=False, iters=8, batch_size=5, verbose=False,
                    n_starts=4, design=design, seed=53,
                    start_bounds={"categorical_corr": (0.1, 0.2)},
                )
                results = gp.multistart_results
                self.assertEqual(len(results), 4)
                winner = min(results, key=lambda row: row["loss"])
                self.assertTrue(winner["selected"])
                self.assertEqual(sum(row["selected"] for row in results), 1)
                self.assertAlmostEqual(hp["categorical_corr"], winner["fitted"]["categorical_corr"])
                self.assertAlmostEqual(gp._training_loss(5), winner["loss"], places=9)
                for row in results:
                    for phase in ("initial", "fitted"):
                        self.assertAlmostEqual(row[phase]["varnoise"], 0.02)
                        np.testing.assert_allclose(row[phase]["lengthscale"], [0.6])
                self.assertTrue(all(0.1 <= row["initial"]["categorical_corr"] <= 0.2 for row in results[1:]))
                self.assertTrue(any(row["initial"]["categorical_corr"] != row["fitted"]["categorical_corr"]
                                    for row in results))
                self.assertTrue(torch.isfinite(gp.get_cond_mean(self.x)).all())

    def test_fixed_rho_and_category_flags_in_design(self):
        gp = GP(self.x, self.y, **self.options)
        hp = gp.Optimise_hyperparameters(n_starts=2, design="lhs", seed=2, iters=2,
                                         fix_categorical_corr=True, verbose=False)
        self.assertEqual(hp["rho"], -0.3)
        self.assertAlmostEqual(hp["categorical_corr"], 0.4)
        for row in gp.multistart_results:
            self.assertAlmostEqual(row["initial"]["categorical_corr"], 0.4)
            self.assertEqual(row["initial"]["rho"], -0.3)
        with self.assertRaisesRegex(ValueError, "fixed parameter"):
            gp.Optimise_hyperparameters(n_starts=2, start_bounds={"rho": (-0.8, 0.8)})


if __name__ == "__main__":
    unittest.main()
