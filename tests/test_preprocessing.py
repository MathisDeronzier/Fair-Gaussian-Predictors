import unittest
import torch
from Fair_GPs.Data_Processing import normalize_train_test


class NormalizationTests(unittest.TestCase):
    def test_train_bounds_shared_with_test_and_explicit_clipping(self):
        train = torch.tensor([[0., 5., 0.], [10., 15., 1.]])
        test = torch.tensor([[-5., 10., 1.], [20., 20., 0.]])
        ytrain = torch.tensor([[2.], [4.]])
        ytest = torch.tensor([[3.], [6.]])
        original = train.clone()
        xt, yt, xv, yv, info = normalize_train_test(train, ytrain, test, ytest)
        torch.testing.assert_close(xt[:, :-1], torch.tensor([[0., 0.], [1., 1.]]).double())
        torch.testing.assert_close(xv[:, :-1], torch.tensor([[0., .5], [1., 1.]]).double())
        torch.testing.assert_close(yv, torch.tensor([[.5], [1.]]).double())
        torch.testing.assert_close(xv[:, -1], test[:, -1].double())
        torch.testing.assert_close(train, original)
        self.assertEqual(info['clipping_report'][1]['x_values_clipped'], 3)
        self.assertEqual(info['clipping_report'][1]['y_values_clipped'], 1)

    def test_constant_columns_and_bad_data(self):
        train = torch.tensor([[2., 0.], [2., 1.]])
        y = torch.ones(2)
        xt, yt, _, _, _ = normalize_train_test(train, y, train, y)
        self.assertTrue(torch.isfinite(xt).all())
        self.assertTrue((xt[:, 0] == 0).all())
        self.assertTrue((yt == 0).all())
        with self.assertRaises(ValueError):
            normalize_train_test(train, y, train, torch.ones(3))


if __name__ == '__main__':
    unittest.main()
