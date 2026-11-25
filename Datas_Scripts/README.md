# Dataset preparation and server experiments

`Data_Preprocessing.py` loads and recodes the source datasets. To fit GPs,
run penalty paths, make predictions, and save results, launch
`Run_Datas_Analysis.py`. The [real-data notebook](../notebook/Run_Datas_Analysis.ipynb)
uses the same experiment functions with a small CPU configuration.

Run the commands below from the repository root. Raw files are downloaded
into `Datas_Scripts/data/`; `--data-dir` selects a different raw-data directory.
The repository does not contain the raw datasets.

## Install and check the workflow

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python Datas_Scripts/download_datasets.py

python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_smoke --datasets law --splits 1 \
  --smoke --lambda-count 2 --workers 1 --threads 2
```

This check fits all six rho models and runs MMD and Sinkhorn. `--smoke`
uses three GP iterations per start, one generated start plus the baseline,
and two optimization iterations per positive penalty. Its default caps are
100 training and 200 test observations. It is an execution check, not a
convergence study. Pass `--splits 1` explicitly for a single split; otherwise
smoke mode uses two seeds. Smoke mode overrides `--gp-iters`, `--doe-points`,
`--mmd-iters`, and `--sinkhorn-iters`, but preserves an explicitly supplied
observation cap and the chosen lambda grid.

Use `--plan` to inspect effective settings and workload without training:

```bash
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/crime_law --datasets crime law --splits 10 \
  --workers 4 --threads 2 --plan
```

Remove `--plan` when the printed configuration is the one you want to run.

## Download the five datasets

The datasets are obtained from their original publishers or the authors'
research repository. The helper downloads and adapts them to the filenames
and schemas consumed by `Data_Preprocessing.py`:

```bash
# Download all five datasets to the default local data directory.
python Datas_Scripts/download_datasets.py

# Download only the datasets needed for an experiment.
python Datas_Scripts/download_datasets.py --datasets crime law

# Choose an external data directory and check its prepared files.
python Datas_Scripts/download_datasets.py --datasets crime law --data-dir /path/to/raw-data
python Datas_Scripts/download_datasets.py --datasets crime law --data-dir /path/to/raw-data --check-only
```

Existing prepared files are kept unless `--force` is supplied. Use
`--check-only` to validate local files without downloading replacements.
When an experiment uses an external directory, give that same path to the
runner's `--data-dir` option.

| Dataset | Brief description | Source | Prepared files and format |
|---|---|---|---|
| `crime` | Community attributes used to predict `ViolentCrimesPerPop`; the sensitive group is derived from the Black population fraction. | [UCI Communities and Crime](https://archive.ics.uci.edu/dataset/183/communities+and+crime) | `communities.data`: headerless comma-separated data; `communities.names`: source metadata; `attributes.csv`: one `attributes` column containing the 128 field names in source order. |
| `compas` | Recidivism records with `two_year_recid` as target; the reader retains the African-American and Caucasian groups. | [ProPublica COMPAS analysis](https://github.com/propublica/compas-analysis) | `compas-scores-two-years.csv`: CSV with its original header, including `race`, `age`, juvenile/prior counts, `event`, `decile_score`, and `two_year_recid`. |
| `student` | Portuguese secondary-school performance, predicting final grade `G3`, with sex as the sensitive attribute. | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | Extract `student-por.csv` from the nested archive and rename it `student.csv`; preserve its header and semicolon delimiter. |
| `law` | Law-school results, predicting `ZFYA` from `LSAT` and `UGPA`, with race as the sensitive attribute. | [Counterfactual Fairness authors](https://github.com/mkusner/counterfactual-fairness) | [`law_data.csv`](https://www.dropbox.com/s/ca74he84aw6ed3z/law_data.csv?dl=1): CSV with a header containing `race`, `LSAT`, `UGPA`, and `ZFYA`; keep the categorical race labels and numeric columns. |
| `adult` | Census income classification, using income above 50K as the target and sex as the sensitive attribute. | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult) | Rename the headerless 15-column `adult.data` and `adult.test` files to `adult.data.csv` and `adult.test.csv`; remove the test file's leading non-data comment line when present. |

The helper derives the Crime field-name file from the `@attribute` lines in
`communities.names`. It uses the Portuguese Student subset, not the separate
Mathematics subset. Adult test-label trailing periods are preserved and
handled by the reader. The COMPAS source is pinned to a repository revision
in the downloader; its exact URL is recorded with the download metadata.
`data/download_manifest.json` records source URLs, downloaded/prepared-file
checksums, transformations, and validation sizes.

For a manual setup, obtain the source archives or CSVs from the links above,
apply the listed filename/format changes, and place the files in one data
directory. The helper's `--check-only` mode checks whether the prepared data
can be read. All subsequent dataset filtering and encoding are performed by
`Data_Preprocessing.py`.

Downloaded files are not versioned in this repository. Source revisions,
row populations, or checksums can differ from files used in earlier
experiments. Keep the downloaded files and their metadata together with the
experiment manifest when reproducing or comparing results.

## Select datasets, splits, and observation caps

The available names are `crime`, `compas`, `student`, `law`, and `adult`.
Omitting `--datasets` runs all five. Dataset names are separated by spaces.

```bash
# Only Crime and Law, with ten split seeds.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/crime_law_10 --datasets crime law --splits 10

# A larger training subset and a smaller test subset.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_caps --datasets law --splits 10 \
  --train-cap 3000 --test-cap 1000

# Use the complete train and test pools; this can require much more memory.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/crime_uncapped --datasets crime --splits 10 \
  --train-cap none --test-cap none
```

For every seed, the runner first creates disjoint 70% training and 30%
testing pools. It then applies each cap inside its respective pool, using
sensitive-group-stratified subsampling. Caps never move observations
between train and test. A cap larger than its pool leaves the full pool in
use. The resulting numbers of used observations need not have a 70/30
ratio. `--splits 10 --first-seed 20` uses seeds 20 through 29.

All datasets use this randomized split procedure. For Adult, the original
train and test files are combined before making these repeated splits.
Law keeps all known race categories: White is `S=1`, and the other known
categories are `S=0`. Missing or unrecognized race labels are rejected. Source versions may contain
different rows, so record the downloaded files and experiment manifest when
comparing results.

## Hyperparameter fitting and multistart

```bash
# Ten generated Sobol starts plus the baseline, with 2,000 iterations each.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_fit20 --datasets law --splits 10 \
  --doe-points 10 --gp-iters 2000 --design sobol --gp-seed 42

# Use a Latin hypercube design instead.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_lhs --datasets law --splits 10 \
  --doe-points 10 --gp-iters 1000 --design lhs --gp-seed 42
```

`--doe-points` counts generated initializations. Each model also receives
its baseline initialization, so five design points mean six fits per
model. The same budget applies to every start. Selection uses the training
marginal likelihood for an exact GP, or the variational ELBO when the
sparse backend is active. The runner evaluates five fixed correlations,
`[-1, -0.9, 0, 0.9, 1]`, and one fitted correlation. These model identities
are preserved across splits even though the fitted rho changes.

For `D` datasets and `R` split seeds, the number of GP fits is
`D * R * 6 * (doe_points + 1)`. At the default five datasets, 30 seeds,
and five design points, that is 5,400 separate fits.

## Lambda range and optimization budgets

Each model has a zero-penalty baseline and `k = --lambda-count` positive
penalties. Let `P` be the precision used in the quadratic objective, and
let `a = --lambda-min-exp`, `b = --lambda-max-exp`. For `k >= 2`,

\[
\lambda_0=0,\qquad
c_j=\exp\!\left(a+\frac{j-1}{k-1}(b-a)\right),\qquad
\lambda_j=\|P\|_{\mathrm F}\,c_j,
\quad j=1,\ldots,k.
\]

The default is eight positive values with exponents from -1 to 9, plus
zero: nine columns in total. The exponentiation uses `exp`, not a power
of ten. Absolute penalties vary with the dataset, split, and rho because
`P` varies. Compare the shared relative factors `c_j` across models.
When `k=1`, only the lower exponent and lower budget endpoint are used.

```bash
# Four positive penalties, a larger maximum penalty, and ten splits.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_lambda11 --datasets law --splits 10 \
  --lambda-count 4 --lambda-min-exp -1 --lambda-max-exp 11

# Change the iteration endpoints independently for the two penalties.
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/law_longer --datasets law --splits 10 \
  --mmd-iters 100 3000 --sinkhorn-iters 100 2000
```

In the first command the relative positive factors are
`exp(-1), exp(3), exp(7), exp(11)`. Increasing only `b` from 9 to 11
multiplies the maximum relative penalty by `exp(2)`.

For each method, iteration budgets increase linearly between the two
requested endpoints and are rounded to integers. The default endpoints
are 20 and 2,400. Each optimization starts from the preceding penalty's
solution. Learning rates decrease geometrically across positive penalties
from `0.005` to `0.0005`; these rates are constant within each individual
optimization. Early stopping can terminate before the requested budget
when the checked relative objective change falls below `5e-4`.

## CPU parallelism and remote execution

`--workers` runs independent dataset/split jobs concurrently. `--threads`
controls CPU threads per worker. A useful starting allocation is
`workers * threads <= allocated CPU cores`; memory must also cover every
concurrent GP. Dense training matrices grow quadratically in the training
cap. Increase workers after verifying that the chosen caps fit in memory.

```bash
python Datas_Scripts/Run_Datas_Analysis.py \
  --output results/all_30 --splits 30 \
  --workers 4 --threads 2
```

On a remote machine, install the environment in that machine's checkout,
then run the same command from its repository root. For a persistent log
outside a batch scheduler:

```bash
mkdir -p logs
nohup python -u Datas_Scripts/Run_Datas_Analysis.py \
  --output results/crime_law_10 --datasets crime law --splits 10 \
  --workers 4 --threads 2 > logs/crime_law_10.log 2>&1 &
```

An example Slurm job, saved in the repository root, is:

```bash
#!/bin/bash
#SBATCH --job-name=fair-gps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=fair-gps-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u Datas_Scripts/Run_Datas_Analysis.py \
  --output results/slurm_crime_law --datasets crime law --splits 10 \
  --workers 4 --threads 2
```

Submit it with `sbatch job.sh`. Adjust memory and wall time to your dataset
caps and fitting budgets. This runner selects the CPU backend.

A copy of the runner stored elsewhere can locate the package explicitly:

```bash
python /path/to/Run_Datas_Analysis.py \
  --repo-root /path/to/Fair-Gaussian-Predictors \
  --data-dir /path/to/raw-data \
  --output /path/to/results/law --datasets law --splits 10
```

Quote paths containing spaces. Each experiment needs its own output
folder. After an interruption, rerun the same scientific command with
`--resume`. Completed splits are checked and skipped. Worker and thread
counts may be changed during resume. Changed code, data, dependencies,
seeds, caps, or scientific settings require a new output folder.

## Parameter reference

Defaults below refer to a normal run; smoke overrides are described above.
Use `--help` for the command-line parser's reference.

| Parameter | Default | Meaning |
|---|---|---|
| `--output PATH` | Required | Experiment directory. |
| `--datasets NAME ...` | All five | Datasets to process. |
| `--splits R` | `30` | Number of consecutive split seeds. |
| `--first-seed SEED` | `0` | First split seed. |
| `--train-cap N` | `2000` | Maximum selected training rows; `none` uses the full pool. |
| `--test-cap N` | `2000` | Maximum selected test rows; `none` uses the full pool. |
| `--workers N` | `1` | Concurrent dataset/split jobs. |
| `--threads N` | `1` | CPU threads per worker. |
| `--gp-iters N` | `1000` | Maximum GP iterations per start. |
| `--doe-points N` | `5` | Generated initializations, in addition to the baseline. |
| `--design NAME` | `sobol` | `sobol` or `lhs`. |
| `--gp-seed SEED` | `42` | Fitting and design seed. |
| `--lambda-count K` | `8` | Positive penalties, plus the zero baseline. |
| `--lambda-min-exp A` | `-1` | Lowest natural-exponential exponent. |
| `--lambda-max-exp B` | `9` | Highest natural-exponential exponent. |
| `--mmd-iters MIN MAX` | `20 2400` | MMD iteration budget endpoints. |
| `--sinkhorn-iters MIN MAX` | `20 2400` | Sinkhorn iteration budget endpoints. |
| `--prediction-jitter VALUE` | `0` | Explicit diagonal shift before pseudoinversion. |
| `--prediction-rtol VALUE` | Automatic | Relative eigenvalue cutoff; see below. |
| `--prediction-atol VALUE` | `0` | Absolute eigenvalue cutoff. |
| `--save-kernel-inverse` | Disabled | Save the MAP pseudoinverse matrix as an extra artifact. |
| `--metrics-only` | Disabled | Skip dense model artifacts, retaining metrics and split records. |
| `--repo-root PATH` | Auto-detected | Repository containing `Fair_GPs/`. |
| `--data-dir PATH` | `Datas_Scripts/data/` | Raw source files. |
| `--resume` | Disabled | Continue the exact matching experiment. |
| `--smoke` | Disabled | Small execution-check budgets. |
| `--plan` | Disabled | Print effective settings without running fits. |

`--metrics-only` and `--save-kernel-inverse` are mutually exclusive.
Pseudoinverse defaults use `N * machine epsilon` as the relative cutoff,
with `N` training rows. The retained eigenvalues have magnitude greater
than `max(atol, rtol * max(abs(eigenvalues)))`. Positive `atol` with no
explicit `rtol` uses an absolute cutoff. The saved inversion diagnostics record the retained rank and cutoff.
The [regression notebook](../notebook/Fair_GPs_Reg_example.ipynb) demonstrates
the effect of projection on incompatible latent solutions.

## Methodology and saved results

Continuous-feature bounds, target mean, and target scale are fitted on the
selected training rows. Both splits use the same normalization and clipping
rules. Continuous features are clipped to `[0,1]`; targets are centered,
divided by their largest absolute training deviation, and clipped to
`[-1,1]`. Categorical codes and the final sensitive-label column are preserved.
The normalized training targets have mean zero to numerical precision.

For every rho, the GP hyperparameters are fitted on these training data.
Noise is then recalibrated using the training error and empirical transport
cost, before forming the quadratic fairness objective. MMD uses a sum of
two Matérn kernels with bandwidths derived from the posterior mean. Sinkhorn
uses `p=2`, `blur=0.05`, and `debias=True`.

MAP prediction uses `K(X,Z) @ K(Z,Z)^+ @ z` with a cached spectral
decomposition shared across all solutions and both methods. Train metrics
are computed on these predictions as well as test metrics. A singular
kernel can project the optimized latent vector, so `solution_KS` and
prediction `KS` can differ.

The experiment directory contains:

| File or directory | Contents |
|---|---|
| `manifest.json` | Effective settings, split seeds, source/data hashes, dependency versions, and invocation. |
| `status.json` | Completion/failure status. |
| `metrics.csv` | Combined rows from completed dataset/split jobs. |
| `law/split_000/complete.json` | Completion marker and metrics checksum for one split. |
| `law/split_000/split.pt` | Selected row IDs, split pools, normalization metadata, and data tensors when enabled. |
| `law/split_000/metrics.csv` | Train/test metrics for every rho, method, and lambda. |
| `law/split_000/models.csv` | Hyperparameters and fit summaries for the six models. |
| `law/split_000/models/rho_1/posterior.pt` | Posterior statistics, training Gram, state dictionaries, fitting records, and inversion diagnostics. |
| `law/split_000/models/rho_1/paths/MMD.pt` | Optimized solutions, logs, coefficients, predictions, and discrepancy diagnostics. |

Other datasets, seeds, rho keys, and Sinkhorn use the same layout.
`--save-kernel-inverse` adds the materialized pseudoinverse under the
historical `kernel_regularized_inv` key in `posterior.pt`. Without this
flag, prediction applies the eigenpairs directly and does not materialize
that inverse matrix.

Metric columns include:

- `MSE`: squared error on the transformed, clipped target scale.
- `MSE_original`: back-transformed predictions compared with the original,
  unmodified target values.
- `KS`: two-sample KS distance between predictions in the observed groups.
- `disp`: mean squared paired difference between predictions at sensitive
  labels 0 and 1, holding the other inputs fixed.
- `cf_mean_abs`, `cf_max_abs`: absolute paired-difference diagnostics.
- `solution_KS`, `solution_MSE`: metrics of the optimized training vector.
- `interpolation_max_abs`: maximum train difference between predictions
  and the optimized vector.

For `D` datasets, `R` seeds, and `k` positive penalties, a complete experiment
has `D * R * 6 * 2 * 2 * (k + 1)` metric rows. The two factors of two are
MMD/Sinkhorn and train/test.

Shared CDF, metric, fair-regression, and exact one-dimensional Wasserstein
helpers are in `Fair_GPs.utils`. The `KS` helper calls SciPy's two-sample
statistic. The `ks_from_empirical_cdfs` helper independently calculates
the same statistic from explicit empirical CDF counts for validation.

## Create HTML tables and download images

After an experiment has completed, pass its output directory to the viewer:

```bash
python Datas_Scripts/view_results.py results/crime_law_10
```

Open `results/crime_law_10/report/comparison.html` in a web browser. The report
shows one table per dataset, method, and train/test split, with rho models in
rows and relative lambda indices in columns. Cells show mean ± sample SD to
three decimal places, using blue for MSE, red for KS, and ochre for disparate
treatment. Counts and variance are retained in `summary.csv`, outside the
cells. A single completed repetition has no estimated SD and is displayed
with a dash.

The HTML **Download images** button downloads the table figures as a ZIP.
The report also saves the files directly:

| Output | Contents |
|---|---|
| `comparison.html` | Portable report, table filters, and the image download button. |
| `summary.csv` | Numeric means, sample SDs, variances, counts, and model/grid metadata. |
| `images/` | Individual table PNG files. |
| `fair_gp_table_images.zip` | ZIP of those same PNG files. |

Choose a report directory or a different paired-difference measure with:

```bash
python Datas_Scripts/view_results.py results/crime_law_10 \
  --output reports/crime_law_10 --disparity cf_mean_abs
```

The `--disparity` choices are `disp` (mean squared paired difference),
`cf_mean_abs`, and `cf_max_abs`. This changes the third metric displayed in
the report; it does not rerun the experiments.

The viewer also accepts a standalone CSV containing per-run metric rows:

```bash
python Datas_Scripts/view_results.py path/to/metrics.csv --output reports/imported
```

If a combined CSV contains several values in an `experiment` column, select
one explicitly:

```bash
python Datas_Scripts/view_results.py path/to/combined_metrics.csv \
  --experiment experiment_A --output reports/experiment_A
```

Directory input checks completed-split markers and checksums. CSV input
uses the rows supplied in that file. The viewer can run without the GP
package; it requires NumPy, pandas, and Matplotlib.

## Inspect results and aggregate repeated splits

```python
from pathlib import Path
import pandas as pd

root = Path("results/crime_law_10")
metrics = pd.read_csv(root / "metrics.csv")
keys = ["dataset", "method", "split", "model_key", "lambda_index"]
summary = metrics.groupby(keys)[["MSE", "KS", "disp"]].agg(["mean", "std"])
summary.to_csv(root / "summary.csv")

cells = pd.DataFrame(index=summary.index)
for metric in ("MSE", "KS", "disp"):
    mean = summary[(metric, "mean")].map(lambda value: f"{value:.3f}")
    sd = summary[(metric, "std")].map(lambda value: f"{value:.3f}")
    cells[metric] = mean + " ± " + sd
print(cells.loc[("law", "MMD", "test")])
```

The standard deviation is the sample SD across completed split seeds,
using denominator `R_completed - 1`. It describes split-to-split variability
and is not a confidence interval. At least two completed splits are needed
to estimate it. Group by `model_key`, not the floating fitted rho value,
so all fitted-rho models contribute to the same row. Likewise, lambda
columns are aligned by relative penalty index, rather than absolute weight.

For library-level postprocessing and plotting examples, see the
[real-data notebook](../notebook/Run_Datas_Analysis.ipynb). Its analysis section
uses `Fair_GPs.Data_Processing.SolutionAnalyzer`.

`Data_Preprocessing.py` is the importable reader module. It is called by the
experiment pipeline; it is not the server experiment launch command.
