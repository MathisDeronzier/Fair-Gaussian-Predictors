#!/usr/bin/env python3
"""Download public datasets and prepare the filenames used by the readers.

Run from any directory. The default destination is Datas_Scripts/data beside
this script. Existing complete datasets are validated and retained. Use
--force to replace them, or --check-only for a local validation without
network access. Reader validation requires the project dependencies.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile


DATASETS = ("crime", "compas", "student", "law", "adult")
COMPAS_COMMIT = "bafff5da3f2e45eca6c2d5055faad269defd135a"
SOURCE_URLS = {
    "crime": "https://archive.ics.uci.edu/static/public/183/communities+and+crime.zip",
    "compas": (
        "https://raw.githubusercontent.com/propublica/compas-analysis/"
        + COMPAS_COMMIT + "/compas-scores-two-years.csv"
    ),
    "student": "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    "law": "https://www.dropbox.com/s/ca74he84aw6ed3z/law_data.csv?dl=1",
    "adult": "https://archive.ics.uci.edu/static/public/2/adult.zip",
}
SOURCE_PAGES = {
    "crime": "https://archive.ics.uci.edu/dataset/183/communities+and+crime",
    "compas": "https://github.com/propublica/compas-analysis",
    "student": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "law": "https://github.com/mkusner/counterfactual-fairness",
    "adult": "https://archive.ics.uci.edu/dataset/2/adult",
}
PREPARED_FILES = {
    "crime": ("communities.data", "communities.names", "attributes.csv"),
    "compas": ("compas-scores-two-years.csv",),
    "student": ("student.csv",),
    "law": ("law_data.csv",),
    "adult": ("adult.data.csv", "adult.test.csv"),
}
FORMAT_NOTES = {
    "crime": "Keep communities.data and communities.names; derive attributes.csv from @attribute names in source order.",
    "compas": "Keep the complete ProPublica CSV, including event, at the pinned repository commit.",
    "student": "Extract student-por.csv from the nested student.zip and rename it student.csv; preserve semicolons.",
    "law": "Keep the CSV linked from the paper authors' README, including race, LSAT, UGPA, and ZFYA.",
    "adult": "Rename adult.data and adult.test to .csv; remove the test file's leading non-data comment line.",
}
MANIFEST_NAME = "download_manifest.json"


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "Fair-Gaussian-Predictors dataset downloader"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _member(content, name):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        try:
            return archive.read(name)
        except KeyError as error:
            raise ValueError(f"The source archive is missing {name}.") from error


def _crime_attributes(content):
    names = []
    for line in content.decode("utf-8").splitlines():
        match = re.match(r"^\s*@attribute\s+(\S+)\s+", line, re.IGNORECASE)
        if match:
            names.append(match.group(1).strip("'\""))
    if not names:
        raise ValueError("communities.names contains no @attribute definitions.")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["attributes"])
    writer.writerows((name,) for name in names)
    return output.getvalue().encode("utf-8")


def _prepare_files(dataset, content):
    if dataset == "crime":
        names = _member(content, "communities.names")
        return {
            "communities.data": _member(content, "communities.data"),
            "communities.names": names,
            "attributes.csv": _crime_attributes(names),
        }
    if dataset == "student":
        inner = _member(content, "student.zip")
        return {"student.csv": _member(inner, "student-por.csv")}
    if dataset == "adult":
        test = _member(content, "adult.test")
        lines = test.splitlines(keepends=True)
        if lines and lines[0].lstrip().startswith(b"|"):
            test = b"".join(lines[1:])
        return {
            "adult.data.csv": _member(content, "adult.data"),
            "adult.test.csv": test,
        }
    return {PREPARED_FILES[dataset][0]: content}


def validate_dataset(dataset, data_dir):
    """Read the prepared files with the experiment's actual dataset loader."""
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    reader = importlib.import_module("Datas_Scripts.Data_Preprocessing")
    # Validation must inspect current files even after an earlier cached read.
    reader._load_dataset.cache_clear()
    data = reader.load_dataset(dataset, data_dir)
    sensitive = data["XS"][:, -1]
    return {
        "rows": len(data["Y"]),
        "input_columns": data["XS"].shape[1],
        "group_0": int((sensitive == 0).sum()),
        "group_1": int((sensitive == 1).sum()),
        "feature_names": data["feature_names"],
        "categorical_dims": list(data["categorical_dims"]),
    }


def _file_metadata(data_dir, filenames):
    return {
        name: {"sha256": _sha256((data_dir / name).read_bytes()),
               "bytes": (data_dir / name).stat().st_size}
        for name in filenames
    }


def prepare_dataset(dataset, data_dir, *, force=False):
    """Download, format, validate and publish one dataset, or retain existing files."""
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    filenames = PREPARED_FILES[dataset]
    complete = all((data_dir / name).is_file() for name in filenames)
    if complete and not force:
        validation = validate_dataset(dataset, data_dir)
        return {
            "status": "existing",
            "source_page": SOURCE_PAGES[dataset],
            "source_url": SOURCE_URLS[dataset],
            "format": FORMAT_NOTES[dataset],
            "files": _file_metadata(data_dir, filenames),
            "validation": validation,
        }

    content = _fetch(SOURCE_URLS[dataset])
    prepared = _prepare_files(dataset, content)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=data_dir) as folder:
        staging = Path(folder)
        for name, value in prepared.items():
            (staging / name).write_bytes(value)
        validation = validate_dataset(dataset, staging)
        # Check every existing file before replacing any part of the dataset.
        for name, value in prepared.items():
            destination = data_dir / name
            if destination.exists() and not force and destination.read_bytes() != value:
                raise FileExistsError(f"{destination} differs from the source. Use --force to replace it.")
        replacements = [name for name in prepared if force or not (data_dir / name).exists()]
        originals = staging / "originals"
        originals.mkdir()
        for name in replacements:
            if (data_dir / name).exists():
                shutil.copyfile(data_dir / name, originals / name)
        published = []
        try:
            for name in replacements:
                os.replace(staging / name, data_dir / name)
                published.append(name)
        except OSError:
            for name in reversed(published):
                if (originals / name).exists():
                    os.replace(originals / name, data_dir / name)
                else:
                    (data_dir / name).unlink(missing_ok=True)
            raise
    return {
        "status": "downloaded",
        "source_page": SOURCE_PAGES[dataset],
        "source_url": SOURCE_URLS[dataset],
        "source_sha256": _sha256(content),
        "source_bytes": len(content),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "format": FORMAT_NOTES[dataset],
        "files": _file_metadata(data_dir, filenames),
        "validation": validation,
    }


def _save_manifest(data_dir, records):
    manifest_path = data_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "datasets": {}}
    for name, record in records.items():
        previous = manifest["datasets"].get(name, {})
        if record["status"] == "existing" and previous.get("files") == record["files"]:
            record = {**previous, **record}
        manifest["datasets"][name] = record
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=data_dir, prefix=".manifest-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Replace existing dataset files.")
    mode.add_argument("--check-only", action="store_true", help="Validate local files without downloads or writes.")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    for name in dict.fromkeys(args.datasets):
        if args.check_only:
            summary = validate_dataset(name, data_dir)
            status = "validated"
        else:
            record = prepare_dataset(name, data_dir, force=args.force)
            # Keep provenance for completed datasets if a later download fails.
            _save_manifest(data_dir, {name: record})
            summary = record["validation"]
            status = record["status"]
        print(f"{name}: {status}; {summary['rows']} rows, "
              f"{summary['input_columns']} input columns, "
              f"groups {summary['group_0']}/{summary['group_1']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
