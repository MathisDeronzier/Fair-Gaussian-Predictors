import unittest
import numpy as np
import torch
from Fair_GPs.Data_Processing import SolutionAnalyzer
from Fair_GPs.Fair_GPs_classe import GP


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.y = torch.tensor([0., 1., 2., 3.])
        self.s = torch.tensor([0., 0., 1., 1.])
        self.path = [self.y[:, None], (self.y + 1)[:, None]]

    def test_legacy_flat_shapes(self):
        a = SolutionAnalyzer(self.path, [0, 1], self.s, self.y)
        result = a.compute_metrics()
        np.testing.assert_allclose(result['mse_mean'], [0, 1])
        np.testing.assert_allclose(result['ks_mean'], [1, 1])
        self.assertTrue(torch.isnan(result['disp_mean']).all())
        equal_groups = torch.tensor([0., 1., 0., 1.])
        result = SolutionAnalyzer([equal_groups], [0], self.s, self.y).compute_metrics()
        np.testing.assert_allclose(result['mse_mean'], [2.])
        np.testing.assert_allclose(result['ks_mean'], [0.])

    def test_grid_and_counterfactuals(self):
        grid = [self.path, self.path]
        cf1 = [[z + 2 for z in self.path], self.path]
        a = SolutionAnalyzer(grid, [[0, 1], [0, 10]], self.s, self.y,
                             rhos=[0, 1], counterfactuals=(grid, cf1))
        result = a.compute_metrics()
        self.assertEqual(len(result), 4)
        np.testing.assert_allclose(result.disp, [4, 4, 0, 0])
        np.testing.assert_allclose(result['lambda'], [0, 1, 0, 10])
        tensor = SolutionAnalyzer(torch.stack([torch.stack(self.path)]*2),
                                  [0, 1], self.s, self.y, rhos=[0, 1])
        self.assertEqual(len(tensor.compute_metrics()), 4)
        row_tensors = SolutionAnalyzer(grid, [torch.tensor([0, 1]), torch.tensor([0, 10])],
                                       self.s, self.y, rhos=[0, 1])
        np.testing.assert_allclose(row_tensors.compute_metrics()["lambda"], [0, 1, 0, 10])

    def test_colored_grid_table(self):
        import matplotlib.pyplot as plt
        a = SolutionAnalyzer([self.path, self.path], [[0, 1], [0, 10]],
                             self.s, self.y, rhos=[0, 1],
                             counterfactuals=([self.path]*2, [self.path]*2))
        fig, ax = a.plot_metrics_table(lambda_factors=[0, 1])
        entries = [obj for obj in ax.texts if ' : ' in obj.get_text()]
        self.assertEqual(len(entries), 2*2*3)
        self.assertEqual({obj.get_color() for obj in entries}, {'#0072B2', '#B2182B', '#8A6500'})
        plt.close(fig)
        with self.assertRaises(ValueError):
            a.plot_metrics_table(lambda_factors=[1, 2])

    def test_runs_and_selection(self):
        a = SolutionAnalyzer([self.path, [z+2 for z in self.path]], [0, 1],
                             [self.s, self.s], [self.y, self.y], is_list=True)
        np.testing.assert_allclose(a.compute_metrics()['mse_mean'], [2, 5])
        np.testing.assert_allclose(a.compute_metrics(1)['mse_mean'], [4, 9])
        np.testing.assert_allclose(a.compute_metrics(1)['mse_std'], [0, 0])
        one = SolutionAnalyzer([self.path], [0, 1], [self.s], [self.y], is_list=True)
        np.testing.assert_allclose(one.compute_metrics()['mse_mean'], [0, 1])
        grid = SolutionAnalyzer([[self.path, self.path]], [0, 1],
                                [self.s], [self.y], is_list=True, rhos=[0, 1])
        self.assertEqual(len(grid.compute_metrics()), 4)

    def test_invalid_axes_and_groups(self):
        for solutions, s in [([self.path[0]], self.s), (self.path, torch.zeros(4))]:
            with self.assertRaises(ValueError):
                SolutionAnalyzer(solutions, [0, 1], s, self.y).compute_metrics()
        with self.assertRaises(ValueError):
            SolutionAnalyzer(self.path, [0, 1], self.s, self.y,
                             counterfactuals=(self.path, [torch.zeros(2)]*2)).compute_metrics()

    def test_update_preserves_mean_and_predictions(self):
        x = torch.tensor([[0., 0.], [1., 0.], [2., 1.], [3., 1.]])
        gp = GP(x, self.y, rho=0.5, device=torch.device('cpu'))
        with torch.no_grad():
            gp.model.mean_module.constant.fill_(7.)
        before = gp.get_cond_mean(x).clone()
        gp.Update_data(x, self.y)
        self.assertEqual(float(gp.model.mean_module.constant), 7.)
        torch.testing.assert_close(before, gp.get_cond_mean(x))


if __name__ == '__main__':
    unittest.main()
