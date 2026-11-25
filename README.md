# Fair Gaussian Predictors

![Version](https://img.shields.io/badge/version-0.2.0-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Gaussian process regression with MMD and Sinkhorn fairness penalties, built with PyTorch and GPyTorch. Explore the tradeoff between prediction accuracy and differences between sensitive-group prediction distributions.

**Maintainer:** [Mathis Deronzier](https://github.com/MathisDeronzier)

**Contributor:** [Elie Odin](https://github.com/elieodin)

## Features

- Continuous and categorical inputs, with fixed or estimated cross-group correlation.
- Sobol and Latin hypercube multistart hyperparameter fitting.
- MMD and Sinkhorn penalty grids, with cached spectral pseudoinverse prediction.
- Train/test comparisons using MSE, KS, and paired sensitive-attribute prediction differences.
- Repeated real-data experiments on several CPU cores, with HTML reports and downloadable figures.

## Installation

Install from the source repository:

```bash
git clone https://github.com/MathisDeronzier/Fair-Gaussian-Predictors.git
cd Fair-Gaussian-Predictors
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The source workflow is tested on Linux with Python 3.12 and CPU execution. Install a Jupyter interface separately if needed to open the notebooks. The multicore server script uses Unix file locks.

## Examples

| Notebook | Demonstration |
|---|---|
| [Synthetic regression](notebook/Fair_GPs_Reg_example.ipynb) | Fit fair GP predictors, vary rho and penalty weights, and compare the resulting metric tables. |
| [Real-data experiments](notebook/Run_Datas_Analysis.ipynb) | Run the server experiment pipeline with a small configuration suitable for a personal computer. |

Open the notebooks from the repository root or its `notebook/` directory. They contain function-usage examples; their short default budgets demonstrate the workflow rather than establish convergence.

## Real-data experiments

The pipeline supports Crime, COMPAS, Student, Law, and Adult. Raw datasets are downloaded separately and are not bundled in the release source tree.

| Script | Purpose |
|---|---|
| [Download datasets](Datas_Scripts/download_datasets.py) | Obtain the source files and adapt their formats for the dataset readers. |
| [Run experiments](Datas_Scripts/Run_Datas_Analysis.py) | Select datasets, split seeds, fitting budgets, penalty grids, observation caps, and CPU parallelism. |
| [View results](Datas_Scripts/view_results.py) | Create HTML comparison tables, mean ± sample SD summaries, and downloadable PNG/ZIP figures. |

The [dataset and server guide](Datas_Scripts/README.md) provides setup instructions, parameter examples, remote execution commands, and the saved-results format.

## Citation and license

For research use, see [CITATION.cff](CITATION.cff) for the software citation and version metadata.

Released under the [MIT license](LICENSE). Dataset sources and terms are documented separately in the [dataset guide](Datas_Scripts/README.md).

## AI assistance

This project was developed with AI assistance from **GPT-6.1 Sol**.
