"""Saved-result aggregation, portable report output, and integrity checks."""

import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from Datas_Scripts import view_results as viewer


def example_records():
    """Two seeded repetitions with deliberately varying fitted rho values."""
    rows = []
    for seed in (0, 1):
        for method in viewer.METHODS:
            for split in viewer.SPLITS:
                for key, rho, fitted in (("rho_1", 1., False),
                                          ("rho_star", .2 + .4 * seed, True),
                                          ("rho_-0.9", -.9, False)):
                    for index in (2, 0, 1):
                        factor = (0., np.exp(-1.), np.exp(1.))[index]
                        rows.append({"dataset": "law", "run": seed, "method": method,
                                     "split": split, "model_key": key, "rho": rho,
                                     "rho_estimated": fitted, "lambda_index": index,
                                     "lambda": factor * (10 + seed), "lambda_relative": factor,
                                     "MSE": seed + index / 10., "KS": .2 + .1 * seed + .01 * index,
                                     "disp": .05 + .02 * seed + .005 * index})
    return pd.DataFrame(rows)


def write_completed_experiment(root, records):
    manifest = {"experiment_id": "example-test", "datasets": ["law"]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for seed, rows in records.groupby("run"):
        directory = root / "law" / f"split_{seed:03d}"
        directory.mkdir(parents=True)
        metrics = directory / "metrics.csv"
        rows.to_csv(metrics, index=False)
        marker = {"experiment_id": manifest["experiment_id"], "metric_rows": len(rows),
                  "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest()}
        (directory / "complete.json").write_text(json.dumps(marker), encoding="utf-8")


class ViewResultsTests(unittest.TestCase):
    def test_two_split_mean_sample_sd_and_fitted_model_identity(self):
        records = viewer.validate_grid(viewer.validate_records(example_records(), "fixture", "disp"))
        summary = viewer.summarize(records)
        self.assertEqual(len(summary), 36)
        cell = summary[(summary.model_key == "rho_star") & (summary.method == "MMD")
                       & (summary.split == "train") & (summary.lambda_index == 1)].iloc[0]
        self.assertAlmostEqual(cell.MSE_mean, .6)
        self.assertAlmostEqual(cell.MSE_std, np.sqrt(.5))
        self.assertAlmostEqual(cell.MSE_var, .5)
        self.assertAlmostEqual(cell.KS_mean, .26)
        self.assertAlmostEqual(cell.KS_std, np.sqrt(.005))
        self.assertAlmostEqual(cell.disp_mean, .065)
        self.assertEqual(cell.n_splits, 2)
        self.assertAlmostEqual(cell.row_rho, .4)
        self.assertAlmostEqual(cell.rho_min, .2)
        self.assertAlmostEqual(cell.rho_max, .6)
        rows = summary[(summary.method == "MMD") & (summary.split == "train")]
        self.assertEqual(rows.model_key.drop_duplicates().tolist(), ["rho_-0.9", "rho_star", "rho_1"])
        for _, path in rows.groupby("model_key"):
            self.assertEqual(path.lambda_index.tolist(), [0, 1, 2])
        document = viewer.table_html(rows, "disp")
        self.assertLess(document.index("λ<sub>0"), document.index("λ<sub>1"))
        self.assertLess(document.index("λ<sub>1"), document.index("λ<sub>2"))
        self.assertIn("0.600 ± 0.707", document)
        self.assertNotIn("n=", document)
        self.assertNotIn("variance", document)

    def test_standalone_csv_requires_an_explicit_experiment_when_mixed(self):
        first = example_records().assign(experiment="first")
        second = example_records().assign(experiment="second", MSE=3.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            pd.concat([first, second]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Choose one experiment"):
                viewer.read_csv_results(path, "disp", None)
            selected, metadata = viewer.read_csv_results(path, "disp", "second")
            self.assertEqual(len(selected), len(second))
            self.assertTrue(selected.MSE.eq(3.).all())
            self.assertEqual(metadata["experiment"], "second")

    def test_completed_split_checksum_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = example_records()
            write_completed_experiment(root, records)
            actual, metadata = viewer.read_results(root, "disp")
            self.assertEqual(len(actual), len(records))
            self.assertEqual(metadata["experiment_id"], "example-test")
            path = root / "law" / "split_000" / "metrics.csv"
            path.write_bytes(path.read_bytes().replace(b"MMD", b"XYZ", 1))
            with self.assertRaisesRegex(ValueError, "checksum"):
                viewer.read_results(root, "disp")

    def test_portable_html_pngs_and_download_archive(self):
        summary = viewer.summarize(example_records())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded, count = viewer.make_images(summary, {"datasets": ["law"]}, "disp", root)
            self.assertEqual(count, 4)
            images = sorted((root / "images").glob("*.png"))
            self.assertEqual(len(images), 4)
            archive = root / "fair_gp_table_images.zip"
            self.assertEqual(archive.read_bytes(), base64.b64decode(encoded))
            with ZipFile(io.BytesIO(base64.b64decode(encoded))) as compressed:
                self.assertEqual(sorted(compressed.namelist()), sorted(path.name for path in images))
                for name in compressed.namelist():
                    self.assertTrue(compressed.read(name).startswith(b"\x89PNG\r\n\x1a\n"))
            document = viewer.report_html(summary, {"datasets": ["law"]}, "disp", encoded)
            self.assertIn("Fair GP experiment results", document)
            self.assertEqual(document.count("<table>"), 4)
            self.assertIn("Download images", document)
            self.assertIn("data:application/zip;base64,", document)
            self.assertNotIn("https://", document)
            self.assertIn(viewer.COLORS["mse"], document)
            self.assertIn(viewer.COLORS["ks"], document)
            self.assertIn(viewer.COLORS["disp"], document)

    def test_single_split_supports_available_train_or_test_tables(self):
        records = example_records()
        for split in viewer.SPLITS:
            selected = records[(records.run == 0) & (records.method == "MMD") & (records.split == split)]
            summary = viewer.summarize(selected)
            self.assertTrue(summary.MSE_std.isna().all())
            self.assertIn("—", viewer.table_html(summary, "disp"))
            with tempfile.TemporaryDirectory() as directory:
                _, count = viewer.make_images(summary, {}, "disp", Path(directory))
                self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
