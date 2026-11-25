import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Datas_Scripts.Data_Preprocessing import load_dataset, read_and_treat_law_data
from Fair_GPs.Data_Processing import _prepare_split, _select_subset


class DatasetReaderTests(unittest.TestCase):
    def test_missing_dataset_files_explain_the_download_command(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "download_datasets.py"):
                load_dataset("law", folder)

    def test_law_keeps_every_known_race_and_groups_white_against_nonwhite(self):
        races = ["White", "Black", "Asian", "Hispanic", "Other", "Amerindian",
                 "Mexican", "Puertorican", " white ", " black "]
        frame = pd.DataFrame({
            "race": races,
            "LSAT": np.arange(len(races)) + 20,
            "UGPA": np.arange(len(races)) / 10 + 2,
            "ZFYA": np.arange(len(races)) / 5 - 1,
        })
        expected_groups = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        with tempfile.TemporaryDirectory() as folder:
            frame.to_csv(Path(folder) / "law_data.csv", index=False)
            x, y, groups, names = read_and_treat_law_data(
                DirectoryName=folder, SensitiveVarName="race", display=False,
            )
            self.assertEqual(len(y), len(frame))
            np.testing.assert_array_equal(groups, expected_groups)
            np.testing.assert_array_equal(x[:, names.index("race")], expected_groups)
            np.testing.assert_allclose(np.asarray(y, dtype=float).reshape(-1), frame["ZFYA"])

            data = load_dataset("law", folder)
            self.assertEqual(data["feature_names"], ["LSAT", "UGPA", "Sensitive"])
            self.assertEqual(data["categorical_dims"], ())
            np.testing.assert_allclose(data["XS"][:, :-1], frame[["LSAT", "UGPA"]])
            np.testing.assert_array_equal(data["XS"][:, -1], expected_groups)
            np.testing.assert_allclose(data["Y"], frame["ZFYA"])
            np.testing.assert_array_equal(data["row_ids"], np.arange(len(frame)))
            for key in ("XS", "Y", "row_ids"):
                self.assertFalse(data[key].flags.writeable)

    def test_law_rejects_missing_or_unrecognized_race(self):
        for race in (None, "", "Unknown"):
            with self.subTest(race=race), tempfile.TemporaryDirectory() as folder:
                pd.DataFrame({
                    "race": ["White", "Black", race],
                    "LSAT": [30, 31, 32],
                    "UGPA": [3.0, 3.1, 3.2],
                    "ZFYA": [-0.1, 0.2, 0.3],
                }).to_csv(Path(folder) / "law_data.csv", index=False)
                with self.assertRaisesRegex(ValueError, "[Rr]ace"):
                    read_and_treat_law_data(DirectoryName=folder)
                with self.assertRaisesRegex(ValueError, "[Rr]ace"):
                    load_dataset("law", folder)

    def test_compas_loading_preserves_the_existing_filter_and_feature_order(self):
        frame = pd.DataFrame({
            "race": ["African-American", "Caucasian", "Hispanic", "Caucasian"],
            "age": [23, 31, 46, 29],
            "juv_misd_count": [0, 1, 2, 3],
            "juv_other_count": [1, 2, 3, 4],
            "priors_count": [3, 1, 0, 2],
            "event": [7, 99, 7, 7],
            "decile_score": [4, 5, 6, 7],
            "two_year_recid": [0, 1, 0, 1],
        })
        with tempfile.TemporaryDirectory() as folder:
            frame.to_csv(Path(folder) / "compas-scores-two-years.csv", index=False)
            data = load_dataset("compas", folder)
        features = ["age", "juv_misd_count", "juv_other_count", "priors_count",
                    "event", "decile_score"]
        self.assertEqual(data["feature_names"], features + ["Sensitive"])
        self.assertEqual(data["categorical_dims"], (4,))
        np.testing.assert_array_equal(data["XS"][:, :-1], frame.loc[[0, 1, 3], features])
        np.testing.assert_array_equal(data["XS"][:, -1], [0, 1, 1])
        np.testing.assert_array_equal(data["Y"], [0, 1, 1])
        np.testing.assert_array_equal(data["row_ids"], [0, 1, 2])


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        rows = np.arange(60)
        self.data = {
            "XS": np.column_stack((3 * rows + 2, np.resize([-1., 7., 99.], 60),
                                   rows % 2)).astype(float),
            "Y": (rows / 13. - 1) ** 2,
            "row_ids": 1000 + 7 * rows,
            "feature_names": ["continuous", "category", "Sensitive"],
            "categorical_dims": (1,),
        }

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_split_is_reproducible_disjoint_capped_and_matches_previous_seed_order(self):
        original = {key: self.data[key].copy() for key in ("XS", "Y", "row_ids")}
        split = _prepare_split(self.data, seed=11, train_cap=18, test_cap=10)
        repeated = _prepare_split(self.data, seed=11, train_cap=18, test_cap=10)
        self.assert_nested_equal(split, repeated)

        # These selected row IDs preserve the runner's previous seeded behavior.
        self.assertEqual(split["train_indices"].tolist(), [
            1238, 1140, 1126, 1070, 1196, 1224, 1028, 1112, 1371,
            1021, 1399, 1315, 1245, 1119, 1175, 1049, 1385, 1343,
        ])
        self.assertEqual(split["test_indices"].tolist(), [
            1147, 1378, 1301, 1063, 1014, 1406, 1322, 1154, 1294, 1035,
        ])
        train_pool = set(split["train_pool_indices"].tolist())
        test_pool = set(split["test_pool_indices"].tolist())
        self.assertEqual((len(train_pool), len(test_pool)), (42, 18))
        self.assertFalse(train_pool & test_pool)
        self.assertEqual(train_pool | test_pool, set(self.data["row_ids"]))
        self.assertTrue(set(split["train_indices"].tolist()) <= train_pool)
        self.assertTrue(set(split["test_indices"].tolist()) <= test_pool)
        self.assertEqual(len(set(split["train_indices"].tolist())), 18)
        self.assertEqual(len(set(split["test_indices"].tolist())), 10)
        groups = split["XS_train"][:, -1]
        self.assertTrue(bool((groups[:-1] <= groups[1:]).all()))
        for name in ("train", "test"):
            self.assertEqual(set(split[f"XS_{name}"][:, -1].tolist()), {0., 1.})
        for key in original:
            np.testing.assert_array_equal(self.data[key], original[key])

    def test_scaling_uses_only_selected_train_and_keeps_categories_and_labels(self):
        split = _prepare_split(self.data, seed=11, train_cap=18, test_cap=10)
        train_rows = (split["train_indices"].numpy() - 1000) // 7
        train_x = torch.from_numpy(self.data["XS"][train_rows])
        train_y = torch.from_numpy(self.data["Y"][train_rows])
        mean = train_y.mean()
        scale = (train_y - mean).abs().max()
        lower, upper = train_x[:, 0].min(), train_x[:, 0].max()
        self.assert_nested_equal(split["normalization"]["y_mean"], mean)
        self.assert_nested_equal(split["normalization"]["y_scale"], scale)

        for name in ("train", "test"):
            rows = (split[f"{name}_indices"].numpy() - 1000) // 7
            x = torch.from_numpy(self.data["XS"][rows])
            y = torch.from_numpy(self.data["Y"][rows])
            torch.testing.assert_close(split[f"XS_{name}"][:, 0],
                                       ((x[:, 0] - lower) / (upper - lower)).clamp(0, 1))
            torch.testing.assert_close(split[f"XS_{name}"][:, 1:], x[:, 1:])
            torch.testing.assert_close(split[f"Y_{name}"], ((y - mean) / scale).clamp(-1, 1))
            torch.testing.assert_close(split[f"Y_{name}_original"], y)
        self.assertAlmostEqual(float(split["Y_train"].mean()), 0., places=14)

        changed = {**self.data, "XS": self.data["XS"].copy(), "Y": self.data["Y"].copy()}
        test_rows = (split["test_pool_indices"].numpy() - 1000) // 7
        changed["XS"][test_rows, 0] = 1e9
        changed["Y"][test_rows] = 1e9
        changed_split = _prepare_split(changed, seed=11, train_cap=18, test_cap=10)
        self.assert_nested_equal(changed_split["XS_train"], split["XS_train"])
        self.assert_nested_equal(changed_split["Y_train"], split["Y_train"])
        self.assert_nested_equal(changed_split["normalization"]["y_mean"], mean)
        self.assert_nested_equal(changed_split["normalization"]["y_scale"], scale)
        self.assertTrue(bool((changed_split["XS_test"][:, 0] == 1).all()))
        self.assertTrue(bool((changed_split["Y_test"] == 1).all()))

    def test_uncapped_splits_use_entire_pools_and_missing_groups_fail(self):
        split = _prepare_split(self.data, seed=11, train_cap=None, test_cap=None)
        self.assertEqual(len(split["train_indices"]), 42)
        self.assertEqual(len(split["test_indices"]), 18)
        self.assertEqual(set(split["train_indices"].tolist()),
                         set(split["train_pool_indices"].tolist()))
        self.assertEqual(set(split["test_indices"].tolist()),
                         set(split["test_pool_indices"].tolist()))
        with self.assertRaisesRegex(ValueError, "lacks a sensitive group"):
            _select_subset(np.arange(5), np.zeros(5), cap=None, seed=11)


if __name__ == "__main__":
    unittest.main()
