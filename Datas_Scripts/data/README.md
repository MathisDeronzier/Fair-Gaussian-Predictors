# Local dataset files

This directory contains local downloads used by the dataset readers. The raw
datasets are not included in the Git repository.

From the repository root, download and prepare the files with:

```bash
python Datas_Scripts/download_datasets.py
```

To prepare only selected datasets or validate existing files:

```bash
python Datas_Scripts/download_datasets.py --datasets crime law
python Datas_Scripts/download_datasets.py --datasets crime law --check-only
```

Expected prepared filenames are:

| Dataset | Files |
|---|---|
| Crime | `communities.data`, `communities.names`, `attributes.csv` |
| COMPAS | `compas-scores-two-years.csv` |
| Student | `student.csv` |
| Law | `law_data.csv` |
| Adult | `adult.data.csv`, `adult.test.csv` |

The [dataset and server guide](../README.md#download-the-five-datasets) gives
verified source links, brief descriptions, and manual format-adaptation
instructions. `Data_Preprocessing.py` performs the dataset-specific filtering
and encoding after the files have been obtained.

Use `--data-dir /path/to/data` in both the downloader and the experiment
runner to keep the raw files elsewhere. `download_manifest.json` records the source URLs, file hashes, format
transformations, and reader validation sizes. Downloaded data and generated
metadata remain local. Keep their checksums and source information when
reproducing an experiment.
