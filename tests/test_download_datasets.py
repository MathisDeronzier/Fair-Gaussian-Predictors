import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from Datas_Scripts import download_datasets as downloader


def archive_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return stream.getvalue()


LAW_CSV = b"race,LSAT,UGPA,ZFYA\nWhite,30,3.1,0.2\nBlack,27,2.9,-0.1\nAsian,33,3.6,0.5\n"


class DownloadFormattingTests(unittest.TestCase):
    def test_student_selects_portuguese_subject_and_preserves_semicolons(self):
        portuguese = b"school;sex;G3\nGP;F;11\n"
        inner = archive_bytes({
            "student-mat.csv": b"school;sex;G3\nGP;M;7\n",
            "student-por.csv": portuguese,
        })
        outer = archive_bytes({"student.zip": inner, ".student.zip_old": b"unused"})
        self.assertEqual(downloader._prepare_files("student", outer),
                         {"student.csv": portuguese})

    def test_crime_attribute_names_keep_source_order(self):
        names = b"Dataset description\n@attribute state numeric\n@ATTRIBUTE county numeric\n@attribute ViolentCrimesPerPop numeric\n"
        archive = archive_bytes({"communities.names": names,
                                 "communities.data": b"1,2,0.3\n"})
        files = downloader._prepare_files("crime", archive)
        self.assertEqual(files["attributes.csv"],
                         b"attributes\nstate\ncounty\nViolentCrimesPerPop\n")
        self.assertEqual(files["communities.names"], names)
        self.assertEqual(files["communities.data"], b"1,2,0.3\n")
        with self.assertRaisesRegex(ValueError, "no @attribute"):
            downloader._crime_attributes(b"No definitions here")

    def test_adult_removes_only_the_official_test_comment(self):
        train, test = b"train row, <=50K\n", b"test row, >50K.\n"
        for header in (b"", b"|1x3 Cross validator\n"):
            archive = archive_bytes({"adult.data": train, "adult.test": header + test})
            self.assertEqual(downloader._prepare_files("adult", archive),
                             {"adult.data.csv": train, "adult.test.csv": test})


class DownloadWorkflowTests(unittest.TestCase):
    def test_law_download_validates_reader_and_records_source_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder)
            with patch.object(downloader, "_fetch", return_value=LAW_CSV) as fetch:
                record = downloader.prepare_dataset("law", data_dir)
            fetch.assert_called_once_with(downloader.SOURCE_URLS["law"])
            self.assertEqual((data_dir / "law_data.csv").read_bytes(), LAW_CSV)
            self.assertEqual(record["validation"]["rows"], 3)
            self.assertEqual(record["validation"]["input_columns"], 3)
            self.assertEqual(record["validation"]["group_0"], 2)
            self.assertEqual(record["validation"]["group_1"], 1)
            expected_hash = hashlib.sha256(LAW_CSV).hexdigest()
            self.assertEqual(record["source_sha256"], expected_hash)
            self.assertEqual(record["files"]["law_data.csv"]["sha256"], expected_hash)
            with patch.object(downloader, "_fetch", side_effect=AssertionError("Unexpected download")):
                retained = downloader.prepare_dataset("law", data_dir)
            self.assertEqual(retained["status"], "existing")
            self.assertEqual(retained["files"], record["files"])

    def test_failed_validation_preserves_existing_files_even_with_force(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "law_data.csv"
            path.write_bytes(LAW_CSV)
            bad_source = b"race,LSAT,UGPA,ZFYA\nWhite,30,3.1,0.2\nUnknown,20,2,-1\n"
            with patch.object(downloader, "_fetch", return_value=bad_source):
                with self.assertRaisesRegex(ValueError, "race"):
                    downloader.prepare_dataset("law", folder, force=True)
            self.assertEqual(path.read_bytes(), LAW_CSV)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ["law_data.csv"])

    def test_partial_download_does_not_overwrite_a_conflicting_file(self):
        source = archive_bytes({
            "adult.data": b"canonical train\n", "adult.test": b"canonical test\n",
        })
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder)
            train = data_dir / "adult.data.csv"
            train.write_bytes(b"local train\n")
            with patch.object(downloader, "_fetch", return_value=source), patch.object(
                downloader, "validate_dataset", return_value={"rows": 2},
            ):
                with self.assertRaisesRegex(FileExistsError, "--force"):
                    downloader.prepare_dataset("adult", data_dir)
            self.assertEqual(train.read_bytes(), b"local train\n")
            self.assertFalse((data_dir / "adult.test.csv").exists())

    def test_check_only_is_offline_and_does_not_write_a_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder)
            (data_dir / "law_data.csv").write_bytes(LAW_CSV)
            before = {path.name: path.read_bytes() for path in data_dir.iterdir()}
            with patch.object(downloader, "_fetch", side_effect=AssertionError("Network used")):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = downloader.main(["--datasets", "law", "--data-dir", folder, "--check-only"])
            self.assertEqual(result, 0)
            self.assertEqual({path.name: path.read_bytes() for path in data_dir.iterdir()}, before)

    def test_rechecking_existing_files_retains_download_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            arguments = ["--datasets", "law", "--data-dir", folder]
            with patch.object(downloader, "_fetch", return_value=LAW_CSV):
                with contextlib.redirect_stdout(io.StringIO()):
                    downloader.main(arguments)
            manifest_path = Path(folder) / downloader.MANIFEST_NAME
            first = json.loads(manifest_path.read_text())["datasets"]["law"]
            with patch.object(downloader, "_fetch", side_effect=AssertionError("Network used")):
                with contextlib.redirect_stdout(io.StringIO()):
                    downloader.main(arguments)
            second = json.loads(manifest_path.read_text())["datasets"]["law"]
            self.assertEqual(second["source_sha256"], first["source_sha256"])
            self.assertEqual(second["retrieved_at"], first["retrieved_at"])
            self.assertEqual(second["files"], first["files"])
            self.assertEqual(second["status"], "existing")

    def test_replacement_failure_restores_all_existing_files(self):
        source = archive_bytes({"adult.data": b"new train\n", "adult.test": b"new test\n"})
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder)
            originals = {"adult.data.csv": b"old train\n", "adult.test.csv": b"old test\n"}
            for name, content in originals.items():
                (data_dir / name).write_bytes(content)
            replace = downloader.os.replace

            def fail_second_file(source_path, destination):
                if Path(destination).name == "adult.test.csv":
                    raise OSError("Injected write failure")
                return replace(source_path, destination)

            with patch.object(downloader, "_fetch", return_value=source), patch.object(
                downloader, "validate_dataset", return_value={"rows": 2},
            ), patch.object(downloader.os, "replace", side_effect=fail_second_file):
                with self.assertRaisesRegex(OSError, "Injected write failure"):
                    downloader.prepare_dataset("adult", data_dir, force=True)
            self.assertEqual({p.name: p.read_bytes() for p in data_dir.iterdir()}, originals)

    def test_later_download_failure_keeps_completed_dataset_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(downloader, "_fetch", side_effect=[LAW_CSV, OSError("Offline")]):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(OSError, "Offline"):
                        downloader.main(["--datasets", "law", "adult", "--data-dir", folder])
            manifest = json.loads((Path(folder) / downloader.MANIFEST_NAME).read_text())
            self.assertEqual(list(manifest["datasets"]), ["law"])
            self.assertEqual(manifest["datasets"]["law"]["source_sha256"],
                             hashlib.sha256(LAW_CSV).hexdigest())


if __name__ == "__main__":
    unittest.main()
