"""Prepare data and analyze fair Gaussian predictor outputs.

Preprocessing validates categorical columns, applies training-fitted scaling,
and forms reproducible train/test subsets. Postprocessing compares solution
paths and rho/lambda grids with SolutionAnalyzer and creates metric tables,
CDF plots, and diagnostic figures.
"""

import math
from operator import index
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from .utils import KS, MSE, true_cdf


# =============================================================================
# PREPROCESSING — categorical features, scaling, and reproducible data splits
# =============================================================================

def categorical_columns(n_features, categorical_dims):
    """Validate categorical indices among the features before the S column."""
    columns = tuple(index(dim) for dim in categorical_dims)
    if len(set(columns)) != len(columns) or any(not 0 <= dim < n_features for dim in columns):
        raise ValueError("categorical_dims must contain distinct feature indices before the S column.")
    return columns


def normalize_train_test(xs_train, y_train, xs_test, y_test, *, center_targets=False,
                         categorical_dims=()):
    """Scale continuous features to [0, 1] and preserve category codes and S.

    With center_targets=True, subtract the training target mean and divide
    by the largest absolute training deviation. Clip targets to [-1, 1].
    By default, targets use min-max scaling to [0, 1].
    Both splits use training statistics and the same clipping rules.
    """
    arrays = [torch.as_tensor(v).detach().clone().double()
              for v in (xs_train, y_train, xs_test, y_test)]
    xt, yt, xv, yv = arrays
    for x, y in [(xt, yt), (xv, yv)]:
        if x.ndim != 2 or x.shape[1] < 2 or y.ndim not in (1, 2):
            raise ValueError('Expected inputs [N, spatial dimensions + S] and scalar targets')
        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError('Targets must have shape (N,) or (N,1)')
        if not x.shape[0] or x.shape[0] != y.shape[0]:
            raise ValueError('Inputs and targets must have matching, nonempty rows')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Non-finite data')
        if not ((x[:, -1] == 0) | (x[:, -1] == 1)).all():
            raise ValueError('Last input column must be binary S')
    if xt.shape[1] != xv.shape[1]:
        raise ValueError('Train and test feature dimensions differ')
    categorical_dims = categorical_columns(xt.size(1) - 1, categorical_dims)
    continuous = [dim for dim in range(xt.size(1) - 1) if dim not in categorical_dims]
    xmin, xmax = xt[:, continuous].amin(0), xt[:, continuous].amax(0)
    xspan = torch.where(xmax > xmin, xmax-xmin, torch.ones_like(xmin))
    if center_targets:
        y_offset = yt.mean()
        y_scale = (yt - y_offset).abs().amax()
        y_scale = torch.where(y_scale > 0, y_scale, torch.ones_like(y_scale))
        y_lower = -1.0
        target_metadata = dict(y_mean=y_offset, y_scale=y_scale)
    else:
        ymin, ymax = yt.amin(), yt.amax()
        y_offset = ymin
        y_scale = torch.where(ymax > ymin, ymax-ymin, torch.ones_like(ymin))
        y_lower = 0.0
        target_metadata = dict(y_min=ymin, y_max=ymax, y_span=y_scale)
    report = []
    for split, x, y in [('train', xt, yt), ('test', xv, yv)]:
        features = x[:, continuous]
        outside_x = (features < xmin) | (features > xmax)
        scaled_y = (y - y_offset) / y_scale
        if center_targets:
            outside_y = scaled_y.abs() > 1
        else:
            outside_y = (y < ymin) | (y > ymax)
        report.append(dict(split=split, n=x.shape[0],
            x_values_clipped=int(outside_x.sum()),
            x_rows_clipped=int(outside_x.any(dim=1).sum()),
            y_values_clipped=int(outside_y.sum())))
        x[:, continuous] = ((features-xmin)/xspan).clamp(0, 1)
        y.copy_(scaled_y.clamp(y_lower, 1))
    metadata = dict(x_min=xmin, x_max=xmax, x_span=xspan, clipping_report=report, **target_metadata)
    if categorical_dims:
        metadata.update(categorical_dims=categorical_dims, continuous_dims=tuple(continuous))
    return xt, yt, xv, yv, metadata


def _select_subset(pool, sensitive, cap, seed):
    if cap is None or len(pool) <= cap:
        selected = pool
    else:
        selected, _ = train_test_split(pool, train_size=cap, random_state=seed,
                                       stratify=sensitive[pool])
    if set(sensitive[selected]) != {0.0, 1.0}:
        raise ValueError("The evaluation subset lacks a sensitive group. Increase the observation cap.")
    return selected


def _prepare_split(data, seed, train_cap, test_cap):
    xs, y = data["XS"], data["Y"]
    # The caps select within the original split pools and never move a row across them.
    train_pool, test_pool = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)
    train = _select_subset(train_pool, xs[:, -1], train_cap, (seed + 1) % 2**32)
    test = _select_subset(test_pool, xs[:, -1], test_cap, (seed + 2) % 2**32)
    # Group the selected training rows once. The optimizer can then reuse
    # the same precision tensor at every penalty without permuting it.
    train = train[np.argsort(xs[train, -1], kind="stable")]
    x_train, y_train, x_test, y_test, scaling = normalize_train_test(
        torch.from_numpy(xs[train]), torch.from_numpy(y[train]),
        torch.from_numpy(xs[test]), torch.from_numpy(y[test]),
        center_targets=True, categorical_dims=data["categorical_dims"],
    )
    ids = data["row_ids"]
    return {
        "train_pool_indices": torch.from_numpy(ids[train_pool]),
        "test_pool_indices": torch.from_numpy(ids[test_pool]),
        "train_indices": torch.from_numpy(ids[train]), "test_indices": torch.from_numpy(ids[test]),
        "XS_train": x_train, "Y_train": y_train, "Y_train_original": torch.from_numpy(y[train]),
        "XS_test": x_test, "Y_test": y_test, "Y_test_original": torch.from_numpy(y[test]),
        "feature_names": data["feature_names"], "categorical_dims": data["categorical_dims"],
        "normalization": scaling, "run": seed,
    }


# =============================================================================
# POSTPROCESSING — metric computation and solution visualization
# =============================================================================

def _fmt_sci(v, digits=1):
        """73406920 -> '$7.3\\cdot 10^{7}$'"""
        v = float(v)
        if v == 0:
            return r"$0$"
        e = int(np.floor(np.log10(abs(v))))
        m = v / 10.0**e
        if round(abs(m), digits) >= 10.0:      # avoid '10.0e7'
            m, e = m / 10.0, e + 1
        if -2 <= e <= 2:                       # use plain notation for values of order one
            return rf"${v:.{digits}f}$"
        return rf"${m:.{digits}f}\cdot 10^{{{e}}}$"


class SolutionAnalyzer:
    """Analyze lambda paths, rho/lambda grids and repeated runs.

    With rhos=None, solutions has axes [lambda][N]. With rhos supplied,
    axes are [rho][lambda][N]. is_list=True adds a leading run axis and
    requires one S/y_train vector per run. Solutions may be tensors or
    nested lists of tensors, with each sample vector shaped (N,) or (N,1).
    lambdas is a shared vector (L,) or a grid (R,L).
    counterfactuals=(pred_s0,pred_s1) has the same axes as solutions and
    enables disp, the mean squared difference on matched individuals.
    compute_metrics returns a DataFrame for grids and the historical
    statistics dictionary for legacy paths; as_frame=True always returns
    a DataFrame. delta_dists remains available for legacy plotting.
    """

    def __init__(self, solutions, lambdas, S, y_train, delta_dists=None, is_list=False, rhos=None, counterfactuals=None):
        self.solutions = solutions
        self.lambdas = lambdas
        self.n_lambda = len(lambdas[0]) if len(lambdas) and torch.as_tensor(lambdas[0]).ndim > 0 else len(lambdas)
        self.counterfactuals = counterfactuals
        self.y_train = y_train
        self.S = S
        self.dists = delta_dists
        self.true_cdf = true_cdf
        self.mse_fn = MSE
        self.ks_fn = KS
        self.N = len(S)
        self.is_list = is_list
        self.rhos = rhos
        if is_list:
            self.n_runs = len(solutions)
            self.ind0 = [(s == 0).nonzero(as_tuple=True)[0] for s in S]
            self.ind1 = [(s == 1).nonzero(as_tuple=True)[0] for s in S]
        else:
            self.n_runs = 1
            self.ind0 = (S == 0).nonzero(as_tuple=True)[0]
            self.ind1 = (S == 1).nonzero(as_tuple=True)[0]
            
        #Comparison datasets
        self.BASELINES = {"crime": 
                        {
                                # "RLS":        {"MSE": 0.033, "KS": 0.55},
                                # "RLS+Berk":   {"MSE": 0.037, "KS": 0.16},
                                # "RLS+Oneto":  {"MSE": 0.037, "KS": 0.14},
                                # "RLS+Ours":   {"MSE": 0.041, "KS": 0.12},

                                # "KRLS":        {"MSE": 0.024, "KS": 0.52},
                                "KRLS+Oneto": {"MSE": 0.028, "KS": 0.19},
                                #"KRLS+Perez": {"MSE": 0.033, "KS": 0.25},
                                "KRLS+Chzhen":  {"MSE": 0.034, "KS": 0.09},

                                # "RF":          {"MSE": 0.020, "KS": 0.45},
                                # "RF+Raff":     {"MSE": 0.030, "KS": 0.21},
                                "RF+Agar":     {"MSE": 0.029, "KS": 0.13},
                                "RF+Chzhen":     {"MSE": 0.033, "KS": 0.08},
                        },

                            "law": 
                            {
                                # "RLS":        {"MSE": 0.107, "KS": 0.15},
                                # "RLS+Berk":   {"MSE": 0.121, "KS": 0.10},
                                # "RLS+Oneto":  {"MSE": 0.112, "KS": 0.07},
                                # "RLS+Chzhen":   {"MSE": 0.141, "KS": 0.02},

                                "KRLS":        {"MSE": 0.040, "KS": 0.09},
                                "KRLS+Oneto": {"MSE": 0.046, "KS": 0.05},
                                "KRLS+Perez": {"MSE": 0.048, "KS": 0.04},
                                "KRLS+Chzhen":  {"MSE": 0.056, "KS": 0.02},

                                "RF":          {"MSE": 0.046, "KS": 0.11},
                                "RF+Raff":     {"MSE": 0.058, "KS": 0.06},
                                "RF+Agar":     {"MSE": 0.050, "KS": 0.05},
                                "RF+Chzhen":     {"MSE": 0.064, "KS": 0.02},
                            },

                            "nlsy": {
                                "KRLS":        {"MSE": 0.061, "KS": 0.58},
                                "KRLS+Oneto":  {"MSE": 0.066, "KS": 0.06},
                                "KRLS+Perez":  {"MSE": 0.065, "KS": 0.08},
                                "KRLS+Chzhen": {"MSE": 0.081, "KS": 0.03},

                                "RF":          {"MSE": 0.055, "KS": 0.55},
                                "RF+Raff":     {"MSE": 0.066, "KS": 0.08},
                                "RF+Agar":     {"MSE": 0.065, "KS": 0.07},
                                "RF+Chzhen":   {"MSE": 0.070, "KS": 0.03},
                            },

                            "student": {
                                "KRLS":        {"MSE": 3.85, "KS": 0.47},
                                "KRLS+Oneto":  {"MSE": 4.07, "KS": 0.18},
                                "KRLS+Perez":  {"MSE": 3.97, "KS": 0.14},
                                "KRLS+Chzhen": {"MSE": 4.46, "KS": 0.03},

                                "RF":          {"MSE": 3.59, "KS": 0.45},
                                "RF+Raff":     {"MSE": 4.28, "KS": 0.09},
                                "RF+Agar":     {"MSE": 3.87, "KS": 0.07},
                                "RF+Chzhen":   {"MSE": 4.18, "KS": 0.02},
                            },

                            "univ": {
                                "KRLS":        {"MSE": 1.43, "KS": 0.10},
                                "KRLS+Oneto":  {"MSE": 1.46, "KS": 0.04},
                                "KRLS+Perez":  {"MSE": 1.50, "KS": 0.06},
                                "KRLS+Chzhen": {"MSE": 1.71, "KS": 0.02},

                                "RF":          {"MSE": 1.31, "KS": 0.10},
                                "RF+Raff":     {"MSE": 1.38, "KS": 0.02},
                                "RF+Agar":     {"MSE": 1.40, "KS": 0.02},
                                "RF+Chzhen":   {"MSE": 1.49, "KS": 0.01},
                            }
                        }

    # ----------------------------------------------------------------------
    # Utility
    # ----------------------------------------------------------------------
    def _extract_pair(self, sol):
        """
        Extracts (X, Y) from a solution using the binary selector S:
        S == 0 --> X samples
        S == 1 --> Y samples
        Ensures output is column vectors (n,1).
        """
        sol_cpu = sol.cpu()          # ensure sol is CPU for plotting
        X = sol_cpu[self.ind0].reshape(-1, 1)
        Y = sol_cpu[self.ind1].reshape(-1, 1)
        return X, Y
    
    # ----------------------------------------------------------------------
    # Saving plots
    # ----------------------------------------------------------------------

    def _save_fig(self, fig, fname):
        """
        Save figure to ../../figs if requested.
        """
        fig_dir = Path("../../figs")
        fig_dir.mkdir(parents=True, exist_ok=True)

        path = fig_dir / fname
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"[Saved] {path.resolve()}")

    # ----------------------------------------------------------------------
    # Plotting: multiple subplots
    # ----------------------------------------------------------------------
    def plot_solutions(self, idx_start=0, nr=2, idx_end=None, ind_sol=None, save = False, fname="solutions_grid.pdf"):
        """
        Plot several transport solutions in a grid.
        nr = number of columns.
        """
        if self.rhos is not None:
            raise ValueError("For grids use plot_metrics_grid or plot_cdfs_grid; for legacy plots construct an analyzer for one rho.")

        if idx_end is None and ind_sol is None:
            idx_end = len(self.solutions)
        if idx_end is None and ind_sol is not None:
            idx_end = len(self.solutions[ind_sol])
            
        if ind_sol is not None:
            subset = self.solutions[ind_sol][idx_start:idx_end]
            subset = [sol.cpu() for sol in subset]
        elif ind_sol is None and self.is_list==False:
            subset = self.solutions[idx_start:idx_end]
        else:
            print("Error: please provide ind_sol for non-list solutions.")
            return  
            
        # Extract subset
        subset = [sol.cpu() for sol in subset]
        lambdas = [float(l) for l in self.lambdas[idx_start:idx_end]]

        k = len(subset)                          # number of plots to display
        r = (k + nr - 1) // nr                   # number of rows

        # Adjust width proportionally to number of columns
        fig_width = 6 * nr
        fig_height = 4 * r

        fig, axes = plt.subplots(r, nr, figsize=(fig_width, fig_height))
        axes = np.array(axes).reshape(-1)        # flatten safely

        # -----------------------------------------------
        # Determine global xmin/xmax as in your code
        # -----------------------------------------------
        xmin = torch.stack([sol.min() for sol in subset]).min().item()
        xmax = torch.stack([sol.max() for sol in subset]).max().item()

        # Grid for CDF
        x_grid = torch.linspace(xmin, xmax, 100*self.N)

        # -----------------------------------------------
        # Plot loop
        # -----------------------------------------------
        for ax, sol, lam in zip(axes, subset, lambdas):

            # Extract X,Y using S
            X, Y = self._extract_pair(sol)
            X = X.flatten()
            Y = Y.flatten()

            # Offsets
            offset_X = -0.05
            offset_Y = -0.08

            # Points
            ax.scatter(X, torch.zeros_like(X) + offset_X,
                    color="blue", alpha=0.4)
            ax.scatter(Y, torch.zeros_like(Y) + offset_Y,
                    color="red", alpha=0.4)

            # Empirical CDF
            ax.plot(x_grid, self.true_cdf(x_grid, X),
                    color="blue", lw=2, alpha=0.8)
            ax.plot(x_grid, self.true_cdf(x_grid, Y),
                    color="red", lw=2, alpha=0.8)

            # Titles and axes
            ax.set_title(f"Solution for λ = {math.floor(lam)}")
            ax.set_xlim(xmin, xmax)
            #ax.set_xlabel("Value")
            #ax.set_ylabel("CDF / samples")
            ax.legend()
    
        # Remove unused axes
        for j in range(k, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        if save:
            self._save_fig(fig, fname)
        plt.show()

    # ----------------------------------------------------------------------
    # Plot all CDF curves on a single figure
    # ----------------------------------------------------------------------


    def plot_all_cdfs(self, bary=False, ind_sol=None, save=False,
                    fname="all_cdfs.pdf", n_legend=6, alpha_min=0.25):
        """Plot all empirical CDFs on the same figure, optionally with barycenter of Sol[0]."""
        if self.rhos is not None:
            raise ValueError("For grids use plot_metrics_grid or plot_cdfs_grid; for legacy plots construct an analyzer for one rho.")

        if ind_sol is None and self.is_list is True:
            print("Error: please provide an ind_sol indicating which solution you want to plot.")
            return
        elif ind_sol is not None:
            solutions = self.solutions[ind_sol]
        else:
            solutions = self.solutions

        ns = len(solutions)
        alphas = torch.linspace(0.02, 1.0, ns) ** 2

        # lambda indices included in the legend (uniform subsampling)
        if n_legend is None or n_legend >= ns:
            keep = set(range(ns))
        else:
            keep = set(np.linspace(0, ns - 1, n_legend).round().astype(int).tolist())

        fig, ax = plt.subplots(figsize=(10, 6))
        handles, labels = [], []

        for i, (sol, lam, alpha) in enumerate(zip(solutions, self.lambdas, alphas)):
            sol_cpu = sol.detach().cpu().flatten()
            X, Y = self._extract_pair(sol_cpu)
            X, Y = X.flatten(), Y.flatten()

            X_sorted = torch.sort(X)[0]
            Y_sorted = torch.sort(Y)[0]

            xmin = min(X_sorted.min().item(), Y_sorted.min().item())
            xmax = max(X_sorted.max().item(), Y_sorted.max().item())
            x_grid = torch.linspace(xmin, xmax, 100 * self.N)

            cdf_X = self.true_cdf(x_grid, X_sorted)
            cdf_Y = self.true_cdf(x_grid, Y_sorted)
            if not torch.is_tensor(cdf_X): cdf_X = torch.tensor(cdf_X)
            if not torch.is_tensor(cdf_Y): cdf_Y = torch.tensor(cdf_Y)

            a = float(alpha)
            ax.plot(x_grid.numpy(), cdf_X.numpy(), color="blue", alpha=a)
            ax.plot(x_grid.numpy(), cdf_Y.numpy(), color="red", alpha=a, linestyle="--")

            if i in keep:
                a_leg = max(a, alpha_min)      # sinon les petits alphas sont invisibles
                handles.append((Line2D([], [], color="blue", alpha=a_leg, lw=1.6),
                                Line2D([], [], color="red", alpha=a_leg, lw=1.6, ls="--")))
                labels.append(_fmt_sci(lam))

        if bary:
            # Compute the reference curve
            bary_cdf = 0.5 * (cdf_X + cdf_Y)
            h, = ax.plot(x_grid.numpy(), bary_cdf.numpy(), color="black", lw=2)
            handles.append(h)
            labels.append("Wasserstein barycenter")

        ax.legend(
            handles, labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)},
            title=r"$\lambda$",
            loc="lower right", # bbox_to_anchor=(1.01, 1.0), if want separate legend figure
            fontsize=9, title_fontsize=10,
            handlelength=2.6, handletextpad=0.6,
            labelspacing=0.35, borderpad=0.4, framealpha=0.9,
        )

        ax.set_title("All empirical CDFs for different λ")
        ax.set_xlabel("Value")
        ax.set_ylabel("CDF")
        ax.grid(True)
        fig.subplots_adjust(right=0.80)        # reserve space outside the axes for the legend
        if save:
            fig.savefig(fname, bbox_inches="tight")
        plt.show()


    # ----------------------------------------------------------------------
    # Compute all metrics
    # ----------------------------------------------------------------------
    
    def _vector(self, value, name):
        value = torch.as_tensor(value).detach().cpu().double()
        if value.ndim not in (1, 2) or (value.ndim == 2 and value.shape[1] != 1):
            raise ValueError(f"{name}: expected (N,) or (N, 1)")
        value = value.reshape(-1)
        if not value.numel() or not torch.isfinite(value).all():
            raise ValueError(f"{name}: empty or non-finite data")
        return value

    def _paths(self, values):
        # Explicit axes avoid confusing repeated runs with rho values.
        if self.rhos is not None:
            return values if self.is_list else [values]
        return [[run] for run in values] if self.is_list else [[values]]

    def metrics_table(self, ind_sol=None):
        """One row per (run, rho, lambda). disp needs paired predictions.

        solutions: [lambda][N], [rho][lambda][N], or, with is_list=True,
        [run][lambda][N] / [run][rho][lambda][N]. Tensors also work.
        lambdas: [lambda], or [rho][lambda] when rhos is supplied.
        S/y_train: shared vectors, or one vector per run with is_list=True.
        counterfactuals=(pred_s0, pred_s1): same axes as solutions.
        Missing counterfactuals give NaN, never an invented zero.
        """
        import pandas as pd
        paths = self._paths(self.solutions)
        if not len(paths):
            raise ValueError("At least one run is required")
        nr = len(self.rhos) if self.rhos is not None else 1
        if nr == 0:
            raise ValueError("rhos must not be empty")
        lams = (torch.stack([torch.as_tensor(row).detach().cpu().double() for row in self.lambdas])
                if len(self.lambdas) and torch.as_tensor(self.lambdas[0]).ndim > 0
                else torch.as_tensor(self.lambdas).detach().cpu().double())
        if lams.ndim == 1:
            lams = lams.repeat(nr, 1)
        if lams.ndim != 2 or lams.shape[0] != nr or lams.shape[1] == 0:
            raise ValueError("lambdas must have shape (L,) or (R, L)")
        if not torch.isfinite(lams).all() or (lams < 0).any():
            raise ValueError("lambdas must be finite and nonnegative")
        if self.rhos is not None:
            rho_values = torch.as_tensor(self.rhos).double()
            if rho_values.ndim != 1 or not torch.isfinite(rho_values).all() or (rho_values.abs() > 1).any():
                raise ValueError("rhos must be a vector in [-1, 1]")
        cf = None
        if self.counterfactuals is not None:
            if len(self.counterfactuals) != 2:
                raise ValueError("counterfactuals must be (pred_s0, pred_s1)")
            cf = [self._paths(v) for v in self.counterfactuals]
            if any(len(v) != len(paths) for v in cf):
                raise ValueError("counterfactual run axis does not match solutions")
        rows = []
        for run, grid in enumerate(paths):
            y = self._vector(self.y_train[run] if self.is_list else self.y_train, "y_train")
            s = self._vector(self.S[run] if self.is_list else self.S, "S")
            if s.numel() != y.numel() or not ((s == 0) | (s == 1)).all() or not (s == 0).any() or not (s == 1).any():
                raise ValueError("S must match y_train and contain both binary groups")
            if len(grid) != nr or (cf is not None and any(len(v[run]) != nr for v in cf)):
                raise ValueError("rho axis does not match rhos")
            for ri, path in enumerate(grid):
                if len(path) != lams.shape[1] or (cf is not None and any(len(v[run][ri]) != len(path) for v in cf)):
                    raise ValueError("lambda axis does not match lambdas")
                for li, sol in enumerate(path):
                    z = self._vector(sol, "solution")
                    if z.numel() != y.numel():
                        raise ValueError("solution and y_train lengths differ")
                    disp = float("nan")
                    if cf is not None:
                        z0, z1 = [self._vector(v[run][ri][li], "counterfactual") for v in cf]
                        if z0.numel() != y.numel() or z1.numel() != y.numel():
                            raise ValueError("counterfactual predictions must match solutions")
                        disp = float((z0 - z1).square().mean())
                    rows.append(dict(run=run, rho_index=ri,
                        rho=float(self.rhos[ri]) if self.rhos is not None else float("nan"),
                        lambda_index=li, **{"lambda": float(lams[ri, li])},
                        MSE=float((z - y).square().mean()),
                        KS=self.ks_fn(z, s),
                        disp=disp))
        table = pd.DataFrame(rows)
        if ind_sol is not None:
            if not isinstance(ind_sol, int) or not 0 <= ind_sol < len(paths):
                raise ValueError("ind_sol must be a valid run index")
            table = table[table.run == ind_sol].reset_index(drop=True)
        return table

    def compute_metrics(self, ind_sol=None, as_frame=None):
        """Grid -> DataFrame by default; legacy list -> original statistics dict.

        as_frame=True always returns raw per-run rows. Repeated runs are not
        averaged over rho. Standard deviation uses ddof=1, or 0 for one run.
        """
        table = self.metrics_table(ind_sol)
        if as_frame is True or (as_frame is None and self.rhos is not None):
            return table
        if self.rhos is not None:
            raise ValueError("Use as_frame=True for a rho/lambda grid")
        result = {}
        for metric, key in [("MSE", "mse"), ("KS", "ks"), ("disp", "disp")]:
            groups = table.groupby("lambda_index", sort=True)[metric]
            result[key + "_mean"] = torch.tensor(groups.mean().to_numpy())
            result[key + "_std"] = torch.tensor(groups.apply(
                lambda x: x.std(ddof=1) if len(x) > 1 else (0.0 if x.notna().all() else np.nan)
            ).to_numpy())
        return result

    def plot_metrics_table(self, title="Metrics", run=0, lambda_factors=None,
                           digits=4, figsize=None):
        """Rho rows, lambda columns, three labelled/color-coded values per cell.

        Blue=MSE, red=KS, ochre=disp; redundant symbols and labels allow
        interpretation without relying on color. If lambda_factors is given,
        column headers show lambda/||Gamma_inv(rho)||_F. Otherwise lambda
        headers are absolute when shared across rho, or indexed with a
        separate actual-lambda table returned by metrics_table().
        Values are predictions evaluated on the corresponding split; they
        are not clipped here. No aggregation over rho or repeated runs.
        """
        table = self.metrics_table(run)
        nr = int(table.rho_index.nunique())
        nl = int(table.lambda_index.nunique())
        if not isinstance(digits, int) or not 0 <= digits <= 10:
            raise ValueError("digits must be an integer between 0 and 10")
        if lambda_factors is not None:
            factors = np.asarray(lambda_factors, dtype=float)
            if factors.shape != (nl,) or not np.isfinite(factors).all() or (factors < 0).any():
                raise ValueError("lambda_factors must be a finite nonnegative vector of length L")
            actual = table.pivot(index='rho_index', columns='lambda_index', values='lambda').to_numpy()
            positive = np.flatnonzero(factors > 0)
            if len(positive):
                scales = actual[:, positive[0]] / factors[positive[0]]
                if not np.allclose(actual, scales[:, None]*factors):
                    raise ValueError("lambda_factors do not match the lambda grid")
            elif not np.allclose(actual, 0):
                raise ValueError("Zero factors require zero lambdas")
            headers = [f"λ{li}\nc={factor:.3g}" for li, factor in enumerate(factors)]
            subtitle = "Columns: c = λ / ‖Γ⁻¹(ρ)‖F; absolute λ values depend on ρ."
        else:
            shared = table.groupby('lambda_index')['lambda'].nunique().max() == 1
            headers = [f"λ{li}\n{table[table.lambda_index == li]['lambda'].iloc[0]:.3g}"
                       if shared else f"λ{li}" for li in range(nl)]
            subtitle = ("Columns: absolute λ values." if shared else
                        "Columns: λ indices; see metrics_table() for absolute values.")
        colors = {'MSE': '#0072B2', 'KS': '#B2182B', 'disp': '#8A6500'}
        symbols = {'MSE': '■', 'KS': '●', 'disp': '▲'}
        fig, ax = plt.subplots(figsize=figsize or (2.0*nl + 1.5, 1.12*nr + 2.1))
        ax.set_xlim(0, nl+1); ax.set_ylim(nr+1, 0); ax.axis('off')
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((0, 0), nl+1, 1, facecolor='#EAF0F6', edgecolor='none'))
        for ri in range(nr):
            ax.add_patch(Rectangle((0, ri+1), 1, 1, facecolor='#F0F3F6', edgecolor='none'))
            if ri % 2 == 0:
                ax.add_patch(Rectangle((1, ri+1), nl, 1, facecolor='#FAFBFC', edgecolor='none'))
        for x in range(nl+2): ax.plot([x, x], [0, nr+1], color='#C9D0D7', lw=0.7)
        for y in range(nr+2): ax.plot([0, nl+1], [y, y], color='#C9D0D7', lw=0.7)
        ax.text(0.5, 0.5, 'ρ / λ', ha='center', va='center', fontweight='bold', fontsize=12)
        for li, header in enumerate(headers):
            ax.text(li+1.5, 0.5, header, ha='center', va='center', fontsize=11, fontweight='bold')
        for ri in range(nr):
            row = table[table.rho_index == ri]
            rho = row.rho.iloc[0]
            ax.text(0.5, ri+1.5, f'{rho:g}' if np.isfinite(rho) else '—',
                    ha='center', va='center', fontsize=12, fontweight='bold')
            for li in range(nl):
                item = row[row.lambda_index == li].iloc[0]
                for offset, metric in zip([0.24, 0.50, 0.76], colors):
                    value = item[metric]
                    text = f'{value:.{digits}f}' if np.isfinite(value) else 'N/A'
                    if np.isfinite(value) and 0 < abs(value) < 10**(-digits):
                        text = f'{value:.1e}'
                    ax.text(li+1.12, ri+1+offset, f'{symbols[metric]} {metric} : {text}',
                            color=colors[metric], ha='left', va='center', fontsize=10.5,
                            fontweight='medium')
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.035, subtitle + '\n■ MSE: blue    ● KS: red    ▲ disp: ochre — labels and symbols supplement the colors.',
                 ha='center', va='bottom', fontsize=10)
        fig.subplots_adjust(top=0.88, bottom=0.16, left=0.025, right=0.985)
        return fig, ax


    def plot_metrics_grid(self, metric="MSE", run=0):
        """Heatmap: columns are lambda indices (actual lambda may depend on rho)."""
        table = self.metrics_table(run)
        if metric not in ("MSE", "KS", "disp"):
            raise ValueError("metric must be MSE, KS or disp")
        values = table.pivot(index="rho_index", columns="lambda_index", values=metric)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(values, aspect="auto")
        ax.set_yticks(range(len(values)), labels=self.rhos if self.rhos is not None else ["unspecified"])
        ax.set_xticks(range(values.shape[1]))
        ax.set(xlabel="Lambda index (see metrics table)", ylabel="rho", title=metric)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig, ax

    def plot_cdfs_grid(self, rho_index=0, lambda_indices=None, run=0):
        """Exact empirical CDFs for one rho and selected lambda indices."""
        self.metrics_table(run)  # validate data before plotting
        path = self._paths(self.solutions)[run][rho_index]
        s = self._vector(self.S[run] if self.is_list else self.S, "S")
        indices = list(range(len(path))) if lambda_indices is None else list(lambda_indices)
        if not indices:
            raise ValueError("Select at least one lambda index")
        fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), squeeze=False)
        for ax, li in zip(axes[0], indices):
            z = self._vector(path[li], "solution")
            for group, color in [(0, "tab:blue"), (1, "tab:red")]:
                x = torch.sort(z[s == group]).values.numpy()
                ax.step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x)+1)/len(x)],
                        where="post", color=color, label=f"S={group}")
            ax.set(title=f"rho index={rho_index}, lambda index={li}", xlabel="Prediction", ylabel="Empirical CDF")
            ax.legend()
        fig.tight_layout()
        return fig, axes


    # ----------------------------------------------------------------------
    # Plot delta evolution with lambda
    # ----------------------------------------------------------------------
    def plot_penalized_to_constrained(self, save=False, fname="delta_evolution.pdf", dist=None):
        """
        Plot the evolution of the δ distance corresponding to constrained solutions for the penalized solution function for λ.
        """
        if self.rhos is not None:
            raise ValueError("For grids use plot_metrics_grid or plot_cdfs_grid; for legacy plots construct an analyzer for one rho.")

        if dist is None:
            dist = torch.tensor(self.dists)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(torch.sqrt(self.lambdas), 1/dist, color="blue", alpha=0.8)
        plt.title("Evolution of the $1/\\delta$ distance as a function of $\\sqrt{\\lambda}$")
        plt.xlabel(r"$\sqrt{\lambda}$")
        plt.ylabel(r"$\delta^{-1}$ distance")
        plt.grid(True)
        plt.tight_layout()
        if save:
            fig.savefig(fname, bbox_inches="tight")
        plt.show()    

    # ----------------------------------------------------------------------
    # Plot metrics evolution
    # ----------------------------------------------------------------------
    def plot_stat_metrics_paper(
        self,
        compare=False,
        dataset="crime",
        show_legend=False,
        figsize_curve=(4, 4),
        figsize_legend=(4, 4),
        save=False,
        fname_curve="metrics_curves.pdf",
        fname_legend="metrics_legend.pdf"
    ):
        """
        Paper-ready plot:
        - Figure 1: curves only (no legend)
        - Figure 2 (optional): legend only
        """
        if self.rhos is not None:
            raise ValueError("For grids use plot_metrics_grid or plot_cdfs_grid; for legacy plots construct an analyzer for one rho.")


        stats = self.compute_metrics()
        x = np.arange(self.n_lambda)

        # ======================================================
        # STYLES
        # ======================================================
        linestyles = ["-", "--", "-.", ":"]
        baseline_alpha = 0.9
        lw_main = 2.2
        lw_base = 1.6

        # ======================================================
        # FIGURE 1 — CURVES ONLY
        # ======================================================
        fig, ax1 = plt.subplots(figsize=figsize_curve)

        # --- OUR METHOD ---
        l_mse, = ax1.plot(
            x, stats["mse_mean"],
            color="tab:blue",
            lw=lw_main,
            label="MSE (ours)",
        )
        ax1.fill_between(
            x,
            stats["mse_mean"] - stats["mse_std"],
            stats["mse_mean"] + stats["mse_std"],
            color="tab:blue",
            alpha=0.15,
            linewidth=0,
        )

        ax2 = ax1.twinx()

        l_ks, = ax2.plot(
            x, stats["ks_mean"],
            color="tab:red",
            lw=lw_main,
            label="KS (ours)",
        )
        ax2.fill_between(
            x,
            stats["ks_mean"] - stats["ks_std"],
            stats["ks_mean"] + stats["ks_std"],
            color="tab:red",
            alpha=0.15,
            linewidth=0,
        )

        legend_handles = [l_mse, l_ks]
        legend_labels = ["MSE (ours)", "KS (ours)"]

        # ======================================================
        # BASELINES
        # ======================================================
        if compare:
            baselines = self.BASELINES[dataset.lower()]
            methods = list(baselines.keys())

            for i, meth in enumerate(methods):
                ls = linestyles[i % len(linestyles)]

                mse = baselines[meth]["MSE"]
                ks = baselines[meth]["KS"]

                h_mse = ax1.axhline(
                    mse,
                    color="tab:blue",
                    linestyle=ls,
                    lw=lw_base,
                    alpha=baseline_alpha,
                    label=f"MSE – {meth}",
                )

                h_ks = ax2.axhline(
                    ks,
                    color="tab:red",
                    linestyle=ls,
                    lw=lw_base,
                    alpha=baseline_alpha,
                    label=f"KS – {meth}",
                )

                legend_handles.extend([h_mse, h_ks])
                legend_labels.extend([f"MSE – {meth}", f"KS – {meth}"])

        # ======================================================
        # AXES FORMATTING
        # ======================================================
        ax1.set_xlabel(r"Index of $\lambda$")
        ax1.set_ylabel("MSE", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2.set_ylabel("KS", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        
        if dataset=="crime":
            ax1.set_ylim(0.015,0.04)
            ax2.set_ylim(0, 0.6)
            ax1.yaxis.set_major_locator(MultipleLocator(0.005))
        elif dataset=="compas":
            ax1.set_ylim(0,0.6)
            ax2.set_ylim(0, 0.8)
        elif dataset != "crime" and dataset != "compas":
            # --- MSE bounds ---
            mse_min = (stats["mse_mean"] - stats["mse_std"]).min().item()
            mse_max = (stats["mse_mean"] + stats["mse_std"]).max().item()
            mse_pad = 0.1 * (mse_max - mse_min)
            ax1.set_ylim(mse_min - mse_pad, mse_max + mse_pad)
            # --- KS bounds ---
            ks_min = (stats["ks_mean"] - stats["ks_std"]).min().item()
            ks_max = (stats["ks_mean"] + stats["ks_std"]).max().item()
            ks_pad = 0.1 * (ks_max - ks_min)

            ax2.set_ylim(max(0, ks_min - ks_pad), ks_max + ks_pad)
        ax1.grid(True, which="major", linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save:
            self._save_fig(fig, dataset + fname_curve)
        plt.show()

        # ======================================================
        # FIGURE 2 — LEGEND ONLY (OPTIONAL)
        # ======================================================
        if show_legend:
            fig_leg = plt.figure(figsize=figsize_legend)
            fig_leg.legend(
                legend_handles,
                legend_labels,
                loc="center",
                frameon=False,
                fontsize=9,
            )
            plt.axis("off")
            plt.tight_layout()
            
            if save:
                self._save_fig(fig_leg, dataset + fname_legend)
        
            plt.show()
