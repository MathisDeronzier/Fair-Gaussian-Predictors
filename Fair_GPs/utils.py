"""Shared numerical, fairness, and one-dimensional transport utilities.

Contents
--------
1. Tensor conversion
2. Empirical CDFs, MSE, and observed-group KS
3. Matched sensitive-label effects
4. Legacy training-calibration transport
5. Empirical Wasserstein-2 distance and barycenter utilities

The legacy fair_regression function uses torch.quantile linear interpolation
at midpoint rank levels and sample-proportion weights. The
wasserstein2_barycenter_1d function uses the empirical step inverse and
accepts specified group weights. Their transported sample vectors can differ.
Both objective values use the exact empirical squared Wasserstein distance
for their respective weights. The legacy function remains the calibration
entry point used by existing GP experiments.

Importing this module preserves the process's default tensor dtype and
device. Transport grids explicitly use float64, while sample computations
follow the supplied tensors.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import ot
from scipy import stats
import torch

__all__ = [
    "to_cpu", "true_cdf", "MSE", "KS", "ks_from_empirical_cdfs",
    "discriminatory_effect", "fair_regression", "BarycenterResult",
    "empirical_quantile", "w2_squared_1d", "wasserstein2_barycenter_1d",
]


# =============================================================================
# 1. Tensor conversion
# =============================================================================

def to_cpu(t):
    """Detach tensor inputs and move them to CPU; pass other values through."""
    if torch.is_tensor(t):
        return t.detach().cpu()
    return t


# =============================================================================
# 2. Empirical CDFs, MSE, and observed-group KS
# =============================================================================

def true_cdf(xgrid, x):
    """Evaluate the empirical CDF of a vector on an increasing scalar grid."""
    xgrid = to_cpu(xgrid)
    x = to_cpu(x)
    x_sorted = torch.sort(x)[0]
    sol = []
    i = 0
    n = len(x)
    for xi in xgrid:
        if i < n:
            if xi < x_sorted[i]:
                sol.append(i / n)
            else:
                while i < n and xi >= x_sorted[i]:
                    i += 1
                sol.append(i / n)
        else:
            sol.append(1.0)
    return sol


def MSE(y_true, y_pred):
    """Return mean squared error for targets and predictions with matching shapes."""
    y_true = to_cpu(y_true)
    y_pred = to_cpu(y_pred)
    return ((y_true - y_pred) ** 2).mean()


def _binary_group_samples(y, S):
    """Validate scalar observations and binary labels, then return two CPU arrays."""
    y = torch.as_tensor(to_cpu(y), device="cpu")
    sensitive = torch.as_tensor(to_cpu(S), device="cpu")
    for name, values in (("y", y), ("S", sensitive)):
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] != 1):
            raise ValueError(f"{name} must have shape (N,) or (N, 1).")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain finite values.")
    y, sensitive = y.reshape(-1), sensitive.reshape(-1)
    if len(y) != len(sensitive):
        raise ValueError("Predictions and sensitive labels must have the same length.")
    if not ((sensitive == 0) | (sensitive == 1)).all():
        raise ValueError("Sensitive labels must contain only 0 and 1.")
    group0, group1 = sensitive == 0, sensitive == 1
    if not group0.any() or not group1.any():
        raise ValueError("Both sensitive groups must contain at least one observation.")
    return y[group0].numpy(), y[group1].numpy()


def KS(y, S):
    """Return scalar two-sample KS for the observed binary groups.

    Predictions and labels accept vectors shaped (N,) or (N, 1), including
    tensor or array inputs. SciPy computes the two-sample statistic. The
    independent ks_from_empirical_cdfs helper supports numerical audits.
    """
    group0, group1 = _binary_group_samples(y, S)
    return float(stats.ks_2samp(group0, group1).statistic)


def ks_from_empirical_cdfs(values, sensitive):
    """Compute KS directly from the two empirical CDFs for an audit.

    This routine counts sorted samples at the combined support using
    NumPy searchsorted. Predictions and binary labels accept tensor or array
    inputs shaped (N,) or (N, 1), with both groups present.
    """
    group0, group1 = _binary_group_samples(values, sensitive)
    group0 = np.sort(group0)
    group1 = np.sort(group1)
    support = np.unique(np.r_[group0, group1])
    cdf0 = np.searchsorted(group0, support, side='right') / len(group0)
    cdf1 = np.searchsorted(group1, support, side='right') / len(group1)
    return float(np.max(np.abs(cdf0 - cdf1)))


# =============================================================================
# 3. Matched sensitive-label effects
# =============================================================================

def discriminatory_effect(X, gp):
    """Return the maximum absolute matched effect from gp.get_cond_mean.

    The final input column is the binary sensitive label. The comparison
    changes that label for the same spatial observations.
    """
    # Set all sensitive labels to 0 and 1 and compare the mean predictions.
    X_0 = X.clone()
    X_1 = X.clone()
    X_0[:, -1] = 0.0
    X_1[:, -1] = 1.0
    y_pred0 = gp.get_cond_mean(X_0).unsqueeze(1)
    y_pred1 = gp.get_cond_mean(X_1).unsqueeze(1)
    return max(abs(y_pred0 - y_pred1))


# =============================================================================
# 4. Legacy training-calibration transport
# =============================================================================

def fair_regression(y0: torch.tensor, y1: torch.tensor):
    """Return legacy midpoint-rank transports and their calibration cost.

    The group weights are their sample proportions. Cross-group quantiles
    use torch.quantile's linear interpolation at midpoint rank levels.
    The scalar value is p0 * p1 * W2_squared between the empirical samples.
    This entry point preserves the training noise-calibration behavior.
    """
    x, y = y0.reshape(-1), y1.reshape(-1)
    n0, n1 = x.numel(), y.numel()
    p0, p1 = n0 / (n0 + n1), n1 / (n0 + n1)
    xs, ys = torch.sort(x).values, torch.sort(y).values
    rank = lambda z: torch.argsort(torch.argsort(z, stable=True), stable=True)
    uX = ((rank(x).to(torch.float64) + 0.5) / n0).to(y.dtype)
    uY = ((rank(y).to(torch.float64) + 0.5) / n1).to(x.dtype)
    Ty0 = p0 * x + p1 * torch.quantile(ys, uX)        # T_{P^0 -> nu*}(X)
    Ty1 = p1 * y + p0 * torch.quantile(xs, uY)        # T_{P^1 -> nu*}(Y)
    value = p0 * p1 * ot.wasserstein_1d(xs[:, None], ys[:, None], p=2).squeeze(0)
    return (Ty0, Ty1, value)


# =============================================================================
# 5. Empirical Wasserstein-2 distance and barycenter utilities
# =============================================================================

class BarycenterResult(NamedTuple):
    """Output of :func:`wasserstein2_barycenter_1d`.

    Attributes
    ----------
    TX : torch.Tensor
        ``T_{P^0 -> nu*}(X)``, same shape as ``X``. This is a sample of the
        barycenter ``nu*`` (the push-forward of the empirical ``P^0``).
    TY : torch.Tensor
        ``T_{P^1 -> nu*}(Y)``, same shape as ``Y``.
    value : torch.Tensor
        Scalar, the minimum ``p_0 W_2^2(P^0, nu*) + p_1 W_2^2(P^1, nu*)``,
        equal to ``p_0 p_1 W_2^2(P^0, P^1)``.
    """

    TX: torch.Tensor
    TY: torch.Tensor
    value: torch.Tensor


def _flatten_sorted(x: torch.Tensor, name: str) -> torch.Tensor:
    """Validate a sample tensor and return its sorted, flattened version."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.numel() == 0:
        raise ValueError(f"Sample `{name}` must be non-empty.")
    if not torch.isfinite(x).all():
        raise ValueError(f"Sample `{name}` contains non-finite values.")
    return torch.sort(x.reshape(-1)).values


def empirical_quantile(x_sorted: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Left-continuous empirical quantile function of a sorted sample.

    Implements ``Q(u) = inf{x : F_n(x) >= u}`` where ``F_n`` is the empirical
    cdf of ``x_sorted``: for ``u`` in ``((i-1)/n, i/n]``, returns the ``i``-th
    order statistic.

    Parameters
    ----------
    x_sorted : torch.Tensor
        One-dimensional sorted sample of size ``n``.
    u : torch.Tensor
        Levels in ``(0, 1]`` (values are clamped to valid indices).

    Returns
    -------
    torch.Tensor
        ``Q(u)``, same shape as ``u``.
    """
    n = x_sorted.numel()
    idx = torch.ceil(u.to(torch.float64) * n).long().clamp(1, n) - 1
    return x_sorted[idx]


def w2_squared_1d(X: torch.Tensor, Y: torch.Tensor, device = torch) -> torch.Tensor:
    """Exact squared 2-Wasserstein distance between two empirical measures.

    Computes ``W_2^2(P^0_n, P^1_m) = \\int_0^1 (F_0^{-1}(t) - F_1^{-1}(t))^2 dt``
    exactly, by integrating the piecewise-constant quantile functions on the
    merged grid of levels ``{i/n} \\cup {j/m}``. The samples may have
    different sizes.

    Parameters
    ----------
    X, Y : torch.Tensor
        Samples (any shape, flattened internally) of ``P^0`` and ``P^1``.

    Returns
    -------
    torch.Tensor
        Scalar tensor, ``W_2^2`` between the two empirical measures.
    """
    x = _flatten_sorted(X, "X")
    y = _flatten_sorted(Y, "Y")
    n, m = x.numel(), y.numel()
    device = x.device

    # Merged grid of jump levels of both quantile functions, in float64 to
    # make the subsequent index computations numerically safe.
    levels = torch.cat(
        [
            torch.arange(1, n + 1, device=device, dtype=torch.float64) / n,
            torch.arange(1, m + 1, device=device, dtype=torch.float64) / m,
        ]
    )
    t = torch.unique(levels, sorted=True)  # t[-1] == 1.0
    zero = torch.zeros(1, device=device, dtype=torch.float64)
    dt = torch.diff(t, prepend=zero)  # interval lengths, sum to 1
    t_mid = t - dt / 2  # interior points: both quantiles constant there

    diff = empirical_quantile(x, t_mid) - empirical_quantile(y, t_mid)
    return (diff.square() * dt.to(diff.dtype)).sum()


def _midpoint_levels(x_flat: torch.Tensor) -> torch.Tensor:
    """Midpoint empirical-cdf levels ``(rank - 1/2) / n`` in original order."""
    n = x_flat.numel()
    order = torch.argsort(x_flat)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=x_flat.device)
    return (ranks.to(torch.float64) + 0.5) / n


def wasserstein2_barycenter_1d(
    X: torch.Tensor,
    Y: torch.Tensor,
    p0: float,
    p1: float | None = None,
) -> BarycenterResult:
    """Wasserstein-2 barycenter of two one-dimensional empirical measures.

    Solves ``min_nu p_0 W_2^2(P^0, nu) + p_1 W_2^2(P^1, nu)`` for the
    empirical measures supported on ``X`` and ``Y``, and returns the optimal
    transport maps evaluated on the input samples together with the optimal
    value.

    Parameters
    ----------
    X : torch.Tensor
        Sample of ``P^0`` (any shape; the map is applied elementwise).
    Y : torch.Tensor
        Sample of ``P^1`` (any shape, possibly a different size than ``X``).
    p0 : float
        Weight of ``P^0``, in ``[0, 1]``.
    p1 : float, optional
        Weight of ``P^1``. Defaults to ``1 - p0``; if provided, the pair must
        satisfy ``p0 + p1 == 1`` (up to numerical tolerance).

    Returns
    -------
    BarycenterResult
        Named tuple ``(TX, TY, value)`` with
        ``TX = T_{P^0 -> nu*}(X) = p0 * X + p1 * F_1^{-1}(F_0(X))``,
        ``TY = T_{P^1 -> nu*}(Y) = p1 * Y + p0 * F_0^{-1}(F_1(Y))`` and
        ``value = p0 * p1 * W_2^2(P^0, P^1)``.

    Examples
    --------
    >>> import torch
    >>> X = torch.randn(500)          # P^0 ~ N(0, 1)
    >>> Y = 2.0 * torch.randn(300) + 1.0  # P^1 ~ N(1, 4)
    >>> TX, TY, value = wasserstein2_barycenter_1d(X, Y, p0=0.5)
    >>> TX.shape, TY.shape
    (torch.Size([500]), torch.Size([300]))
    """
    if p0 is None and p1 is None:
        p0 = X.shape[0] / (X.shape[0] + Y.shape[0])
        p1 = 1.0 - p0
    elif p1 is None:
        p1 = 1.0 - p0
    if p0 is None or not all(math.isfinite(p) and 0 <= p <= 1 for p in (p0, p1)):
        raise ValueError("Barycenter weights must be finite and lie in [0, 1].")
    if not math.isclose(p0 + p1, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("Barycenter weights must sum to one.")

    x_flat = torch.as_tensor(X).reshape(-1)
    y_flat = torch.as_tensor(Y).reshape(-1)
    x_sorted = _flatten_sorted(X, "X")
    y_sorted = _flatten_sorted(Y, "Y")

    # T_{P^0 -> nu*}(x) = p0 * x + p1 * F_1^{-1}(F_0(x)) and symmetrically.
    TX = p0 * x_flat + p1 * empirical_quantile(y_sorted, _midpoint_levels(x_flat))
    TY = p1 * y_flat + p0 * empirical_quantile(x_sorted, _midpoint_levels(y_flat))

    value = p0 * p1 * w2_squared_1d(X, Y)

    return BarycenterResult(
        TX=TX.reshape(torch.as_tensor(X).shape),
        TY=TY.reshape(torch.as_tensor(Y).shape),
        value=value,
    )
