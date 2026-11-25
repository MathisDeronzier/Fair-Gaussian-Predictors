"""Fair Gaussian predictors and their shared numerical routines.

Contents
--------
1. Tensor conversion and spatial kernel construction
2. Hyperparameter initialization with Sobol or Latin hypercube designs
3. Spectral Moore-Penrose pseudoinverses
4. Continuous, categorical, and sensitive-group covariance kernel
5. Exact and sparse GPyTorch model wrappers
6. Live hyperparameter view
7. GP fitting, conditioning, and cached MAP predictions
8. Batched MAP predictions along a fairness-penalty path

The GP input is [spatial features..., sensitive label], with a binary label
in the final column. One spatial kernel is shared across groups, with
cross-group correlation rho. Categorical features use a learned shared
correlation. The GP class supports exact and sparse fitting. Final MAP
predictions extend latent training solutions through the prior Gram
pseudoinverse, with cached spectral decompositions and coefficients.

The shared routines are available from this module alongside GP.
"""

from copy import deepcopy
from operator import index
from typing import Optional, Union, Iterable, Tuple
import math
import warnings

import gpytorch
import numpy as np
import torch
from scipy.stats import qmc
from linear_operator.operators import DenseLinearOperator
from linear_operator.utils.errors import NanError, NotPSDError

from .Data_Processing import categorical_columns

# Use the same double precision and default device as the GP interface.
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep learned rho inside the endpoints before applying the atanh transform.
_RHO_MAX = 0.999


# =============================================================================
# 1. Tensor conversion and spatial kernel construction
# =============================================================================

def _safe_tensor(x, device: torch.device, dtype=torch.float64,
                 as_long: bool = False, name: str = "tensor") -> torch.Tensor:
    """
    Convert x (np.ndarray / list / tuple / torch.Tensor) to a torch.Tensor on
    `device` with `dtype` (or torch.long if as_long=True). Raises a clear error
    on unsupported types, and rejects NaN/Inf for floating tensors.
    """
    if isinstance(x, np.ndarray):
        try:
            if as_long:
                t = torch.as_tensor(x, dtype=torch.long, device=device)
            else:
                t = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=device)
        except Exception as e:
            raise TypeError(f"Cannot convert numpy array to torch tensor for {name}: {e}")
    elif isinstance(x, torch.Tensor):
        t = x.to(device=device).long() if as_long else x.to(device=device, dtype=dtype)
    elif isinstance(x, (list, tuple)):
        return _safe_tensor(np.asarray(x), device=device, dtype=dtype, as_long=as_long, name=name)
    else:
        raise TypeError(f"Unsupported type for {name}: {type(x)}")

    if not as_long:
        if torch.isnan(t).any():
            raise ValueError(f"{name} contains NaN values after conversion (device={device}).")
        if torch.isinf(t).any():
            raise ValueError(f"{name} contains Inf values after conversion (device={device}).")
    return t


def _make_kernel(name: Union[str, gpytorch.kernels.Kernel],
                 ard_num_dims: Optional[int] = None) -> gpytorch.kernels.Kernel:
    """Return an (un-scaled) GPyTorch kernel from a name or a kernel instance."""
    if isinstance(name, gpytorch.kernels.Kernel):
        return name
    key = name.lower()
    d = ard_num_dims
    if key in ("matern52", "matern", "matern5/2", "matern5_2"):
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
    if key in ("matern32", "matern3/2", "matern3_2"):
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d)
    if key in ("exponential", "exp", "matern12", "matern1/2", "matern1_2"):
        return gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=d)
    if key in ("rbf", "sqexp", "gaussian", "rbfkernel"):
        return gpytorch.kernels.RBFKernel(ard_num_dims=d)
    if key in ("rationalquadratic", "rq"):
        return gpytorch.kernels.RationalQuadraticKernel(ard_num_dims=d)
    return gpytorch.kernels.RBFKernel(ard_num_dims=d)


# =============================================================================
# 2. Hyperparameter initialization with Sobol or Latin hypercube designs
# =============================================================================

def sample_hyperparameter_starts(bounds, count, *, design, seed):
    """Sample lengthscales and variances in log space, other values linearly.

    Each entry in bounds is a lower/upper pair of scalar or equally shaped
    vector bounds. Vector parameters occupy one design axis per component.
    """
    log_scaled = {"lengthscale", "outputscale", "varnoise"}
    axes = []
    lower, upper = [], []
    for name, (lo, hi) in bounds.items():
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if (lo.shape != hi.shape or not np.isfinite(lo).all() or
                not np.isfinite(hi).all() or np.any(lo >= hi)):
            raise ValueError(f"{name} bounds must be finite and increasing with matching shapes.")
        if name in log_scaled:
            if np.any(lo <= 0):
                raise ValueError(f"{name} bounds must be positive.")
            lo, hi = np.log(lo), np.log(hi)
        axes.append((name, lo.shape, slice(len(lower), len(lower) + lo.size)))
        lower.extend(lo.ravel())
        upper.extend(hi.ravel())

    if not axes:
        raise ValueError("A multistart design needs at least one trainable hyperparameter.")
    if design == "sobol":
        engine = qmc.Sobol(d=len(lower), scramble=True, seed=seed)
        unit = engine.random_base2((count - 1).bit_length())[:count]
    elif design == "lhs":
        unit = qmc.LatinHypercube(d=len(lower), seed=seed).random(count)
    else:
        raise ValueError("design must be 'sobol' or 'lhs'.")

    starts = []
    for row in qmc.scale(unit, lower, upper):
        start = {}
        for name, shape, columns in axes:
            value = row[columns].reshape(shape)
            if name in log_scaled:
                value = np.exp(value)
            start[name] = value.item() if not shape else value
        starts.append(start)
    return starts


# =============================================================================
# 3. Spectral Moore-Penrose pseudoinverses
# =============================================================================

def _pseudoinverse_settings(jitter, rtol, atol):
    jitter, atol = float(jitter), float(atol)
    rtol = None if rtol is None else float(rtol)
    if any(not math.isfinite(value) or value < 0 for value in (jitter, atol)):
        raise ValueError("jitter and atol must be finite and nonnegative.")
    if rtol is not None and (not math.isfinite(rtol) or rtol < 0):
        raise ValueError("rtol must be finite and nonnegative, or None.")
    return jitter, rtol, atol


@torch.no_grad()
def symmetric_pinv_factors(matrix, *, jitter=0.0, rtol=None, atol=0.0):
    """Return eigenvectors, reciprocal retained eigenvalues and diagnostics.

    With rtol=None and atol=0, use n * machine epsilon as in torch.linalg.pinv.
    Eigenvalues with magnitude at most max(atol, rtol * max(abs(eigenvalue)))
    are discarded. Positive atol with omitted rtol uses an absolute cutoff.
    jitter is an explicit diagonal addition before decomposition.
    """
    jitter, rtol, atol = _pseudoinverse_settings(jitter, rtol, atol)
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1) or matrix.size(0) == 0:
        raise ValueError("Expected a nonempty square covariance matrix.")
    if not matrix.is_floating_point() or not torch.isfinite(matrix).all():
        raise ValueError("The covariance matrix must contain finite floating-point values.")
    system = (matrix + matrix.T) * 0.5
    if jitter:
        system.diagonal().add_(jitter)
    eigenvalues, vectors = torch.linalg.eigh(system)
    effective_rtol = (len(matrix) * torch.finfo(matrix.dtype).eps if atol == 0 else 0.0) if rtol is None else rtol
    cutoff = max(atol, effective_rtol * float(eigenvalues.abs().max()))
    retained = eigenvalues.abs() > cutoff
    reciprocals = torch.zeros_like(eigenvalues)
    reciprocals[retained] = eigenvalues[retained].reciprocal()
    info = dict(rank=int(retained.sum()), size=len(matrix), cutoff=cutoff,
                effective_rtol=effective_rtol, atol=atol, jitter=jitter)
    return vectors, reciprocals, info


@torch.no_grad()
def apply_symmetric_pinv(vectors, reciprocals, values):
    """Apply the pseudoinverse without materializing an inverse matrix."""
    if values.ndim == 1:
        return vectors @ (reciprocals * (vectors.T @ values))
    if values.ndim != 2:
        raise ValueError("values must have shape (N,) or (N, L).")
    return vectors @ (reciprocals[:, None] * (vectors.T @ values))


@torch.no_grad()
def symmetric_pseudoinverse(matrix, *, jitter=0.0, rtol=None, atol=0.0):
    """Materialize the symmetric Moore-Penrose pseudoinverse."""
    vectors, reciprocals, _ = symmetric_pinv_factors(matrix, jitter=jitter, rtol=rtol, atol=atol)
    inverse = (vectors * reciprocals) @ vectors.T
    return (inverse + inverse.T) * 0.5


# =============================================================================
# 4. Continuous, categorical, and sensitive-group covariance kernel
# =============================================================================

class MixtureKernel(gpytorch.kernels.Kernel):
    """
    k((x, s), (x', s')) = k_base_scaled(x, x') * B[s, s'],  B = [[1, rho],[rho, 1]].

    The last input column is the binary label s in {0, 1}; the remaining columns
    are the spatial part x.

    With categorical_dims, apply the spatial kernel to the continuous columns
    and multiply by q**h, where h is the number of categorical mismatches and
    q is the shared trainable categorical correlation in (0, 1).

    When rho == 1 and rho is not being estimated, `forward` takes a fast path
    that ignores S when there are only continuous features.
    """

    is_stationary = False

    def __init__(self, kernel_x: Union[str, gpytorch.kernels.Kernel] = "matern52",
                 rho: float = 1.0, ard_num_dims: Optional[int] = None,
                 categorical_dims: Iterable[int] = (), categorical_corr: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        # NB: do NOT set self.ard_num_dims here; the outer kernel sees the full
        # input [spatial..., s]. Only the inner spatial kernel uses ARD.
        self.categorical_dims = tuple(categorical_dims)
        if ard_num_dims is not None:
            self.categorical_dims = categorical_columns(ard_num_dims, self.categorical_dims)
        continuous_dims = None if ard_num_dims is None else ard_num_dims - len(self.categorical_dims)
        self.spatial = _make_kernel(kernel_x, ard_num_dims=continuous_dims)
        self.spatial_scaled = gpytorch.kernels.ScaleKernel(self.spatial)
        if self.categorical_dims:
            self.register_parameter("raw_categorical_corr", torch.nn.Parameter(torch.zeros(())))
            self.register_constraint("raw_categorical_corr", gpytorch.constraints.Interval(0.0, 1.0))
            self.categorical_corr = categorical_corr

        rho = float(rho)
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"rho must lie in [-1, 1] (B positive semi-definite); got {rho}.")

        # Two representations, one active at a time (see `learn_rho`).
        dt = torch.get_default_dtype()
        self.learn_rho = False
        self.register_buffer("fixed_rho", torch.tensor(rho, dtype=dt))
        self.register_parameter(
            "raw_rho",
            torch.nn.Parameter(torch.atanh(torch.tensor(np.clip(rho, -_RHO_MAX, _RHO_MAX), dtype=dt))),
        )
        self.raw_rho.requires_grad_(False)

    # -- rho -----------------------------------------------------------------
    def rho_value(self) -> torch.Tensor:
        """Current rho as a tensor (differentiable when it is being estimated)."""
        return torch.tanh(self.raw_rho) if self.learn_rho else self.fixed_rho

    def set_rho(self, value: float):
        """Set rho in [-1, 1]; keeps both representations in sync."""
        value = float(value)
        if not (-1.0 <= value <= 1.0):
            raise ValueError(f"rho must lie in [-1, 1]; got {value}.")
        dt, dev = self.fixed_rho.dtype, self.fixed_rho.device
        with torch.no_grad():
            self.fixed_rho.fill_(value)
            clipped = float(np.clip(value, -_RHO_MAX, _RHO_MAX))
            self.raw_rho.copy_(torch.atanh(torch.tensor(clipped, dtype=dt, device=dev)))
        if abs(value) == 1.0 and self.learn_rho:
            # |rho| = 1 is only exactly representable in the fixed store
            # (tanh never reaches +-1), so asking for it also fixes rho.
            self.learn_rho = False
            self.raw_rho.requires_grad_(False)
            warnings.warn(
                f"rho={value} is exact only when fixed; rho is no longer estimated.",
                RuntimeWarning, stacklevel=2,
            )

    def set_learn_rho(self, learn: bool):
        """Switch rho between 'estimated' and 'fixed', syncing the two stores."""
        learn = bool(learn)
        if learn == self.learn_rho:
            self.raw_rho.requires_grad_(learn)
            return
        dt, dev = self.fixed_rho.dtype, self.fixed_rho.device
        with torch.no_grad():
            if learn:  # fixed -> learned: carry the current value over
                value = float(self.fixed_rho)
                clipped = float(np.clip(value, -_RHO_MAX, _RHO_MAX))
                self.raw_rho.copy_(torch.atanh(torch.tensor(clipped, dtype=dt, device=dev)))
                if clipped != value:
                    warnings.warn(
                        f"rho={value} cannot be estimated exactly (tanh never reaches +-1); "
                        f"starting the optimisation from {clipped}.",
                        RuntimeWarning, stacklevel=2,
                    )
            else:      # learned -> fixed: freeze the value reached
                self.fixed_rho.fill_(float(torch.tanh(self.raw_rho)))
        self.learn_rho = learn
        self.raw_rho.requires_grad_(learn)

    @property
    def is_collapsed(self) -> bool:
        """True when rho == 1 and fixed: s is irrelevant, so B can be skipped."""
        return (not self.learn_rho) and bool(self.fixed_rho == 1)

    @property
    def categorical_corr(self) -> torch.Tensor:
        return self.raw_categorical_corr_constraint.transform(self.raw_categorical_corr)

    @categorical_corr.setter
    def categorical_corr(self, value: float):
        value = float(value)
        if not 0.0 < value < 1.0:
            raise ValueError("categorical_corr must be strictly between 0 and 1.")
        raw = self.raw_categorical_corr
        self.initialize(raw_categorical_corr=self.raw_categorical_corr_constraint.inverse_transform(
            raw.new_tensor(value)
        ))

    def coreg_matrix(self) -> torch.Tensor:
        """The 2x2 matrix B = [[1, rho], [rho, 1]] (differentiable if estimated)."""
        rho = self.rho_value()
        one = torch.ones((), dtype=rho.dtype, device=rho.device)
        return torch.stack([torch.stack([one, rho]), torch.stack([rho, one])])

    # -- split (x, s) --------------------------------------------------------
    def _extract_x_s(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        last = x[..., -1]
        if not torch.all((last == 0) | (last == 1)):
            raise ValueError(
                "The last input column must be the class label s in {0, 1}. "
                "Got values outside {0, 1} (e.g. learned inducing points drifting "
                "off the discrete axis)."
            )
        s = last.long()
        if x.size(-1) > 1:
            x_spatial = x[..., :-1]
        else:
            x_spatial = torch.zeros((x.size(0), 0), device=x.device, dtype=x.dtype)
        return x_spatial, s

    def forward(self, x1, x2, diag=False, **params):
        x1s, s1 = self._extract_x_s(x1)
        x2s, s2 = self._extract_x_s(x2)
        x1s = x1s.to(torch.get_default_dtype())
        x2s = x2s.to(torch.get_default_dtype())

        # Fast path: rho == 1 -> B is all-ones -> the label plays no role at all.
        if self.is_collapsed and not self.categorical_dims:
            return self.spatial_scaled(x1s, x2s, diag=diag, **params)

        same_group = s1 == s2 if diag else s1.unsqueeze(-1) == s2.unsqueeze(-2)
        if self.categorical_dims:
            categorical = categorical_columns(x1s.size(-1), self.categorical_dims)
            continuous = [dim for dim in range(x1s.size(-1)) if dim not in categorical]
            mismatches = torch.zeros_like(same_group, dtype=torch.long)
            for dim in categorical:
                a, b = x1s[..., dim], x2s[..., dim]
                different = a != b if diag else a.unsqueeze(-1) != b.unsqueeze(-2)
                mismatches = mismatches + different
            category_factor = self.categorical_corr.pow(mismatches)
            x1s, x2s = x1s[..., continuous], x2s[..., continuous]
        else:
            category_factor = 1.0

        if x1s.size(-1):
            spatial = self.spatial_scaled(x1s, x2s, diag=diag, **params)
            spatial = spatial if diag else spatial.to_dense()
        else:
            spatial = self.spatial_scaled.outputscale
        rho = self.rho_value()
        cov = spatial * category_factor * torch.where(same_group, torch.ones_like(rho), rho)
        return cov if diag else DenseLinearOperator(cov)


# =============================================================================
# 5. Exact and sparse GPyTorch model wrappers
# =============================================================================

class _ExactGPWrapper(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().double()
        # The kernel already carries the outputscale via spatial_scaled, so we
        # do NOT wrap it in an extra (redundant) ScaleKernel.
        self.covar_module = kernel.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class _SVGPWrapper(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel, learn_inducing_locations: bool = False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        ).double()
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean().double()
        self.covar_module = kernel.double()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# =============================================================================
# 6. Live hyperparameter view
# =============================================================================

class _LiveHyperparameters:
    """
    Live, always-in-sync view of the interpretable hyperparameters.

    Reading returns the *current* torch values (so it reflects any optimisation
    step); assigning writes straight into the model.

    Exposed:
      - varnoise    : observation noise sigma_n^2
      - lengthscale : spatial lengthscale (numpy array; ARD -> length-d vector)
      - outputscale : spatial outputscale sigma_f^2
      - rho         : cross-group correlation in [-1, 1]
      - coreg       : read-only 2x2 matrix B = [[1, rho], [rho, 1]]

    Manual editing:
      hp.varnoise    = 1e-3
      hp.lengthscale = 0.5           # scalar or length-d ARD vector
      hp.outputscale = 2.0
      hp.rho         = 1.0
    """

    def __init__(self, gp: "GP"):
        self._gp = gp

    def _kernel(self) -> MixtureKernel:
        return self._gp.model.covar_module

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, device=self._gp.device, dtype=self._gp.dtype)

    # ---- observation noise ----------------------------------------------
    @property
    def varnoise(self) -> float:
        return float(self._gp.likelihood.noise.detach().cpu().item())

    @varnoise.setter
    def varnoise(self, value):
        self._gp.likelihood.noise = self._tensor(float(value))

    # ---- lengthscale -----------------------------------------------------
    @property
    def lengthscale(self) -> np.ndarray:
        value = self._kernel().spatial.lengthscale
        return np.empty(0) if value is None else value.detach().cpu().numpy().ravel()

    @lengthscale.setter
    def lengthscale(self, value):
        self._kernel().spatial.lengthscale = self._tensor(value)

    # ---- outputscale -----------------------------------------------------
    @property
    def outputscale(self) -> float:
        return float(self._kernel().spatial_scaled.outputscale.detach().cpu().item())

    @outputscale.setter
    def outputscale(self, value):
        self._kernel().spatial_scaled.outputscale = self._tensor(float(value))

    # ---- cross-group correlation ----------------------------------------
    @property
    def rho(self) -> float:
        return float(self._kernel().rho_value().detach().cpu())

    @rho.setter
    def rho(self, value):
        self._kernel().set_rho(value)

    @property
    def categorical_corr(self) -> Optional[float]:
        kernel = self._kernel()
        return float(kernel.categorical_corr.detach()) if kernel.categorical_dims else None

    @categorical_corr.setter
    def categorical_corr(self, value):
        if not self._kernel().categorical_dims:
            raise ValueError("categorical_corr requires categorical_dims.")
        self._kernel().categorical_corr = value

    @property
    def coreg(self) -> np.ndarray:
        """Read-only view of B; set `rho` instead."""
        return self._kernel().coreg_matrix().detach().cpu().numpy()

    def as_dict(self) -> dict:
        values = {"varnoise": self.varnoise, "lengthscale": self.lengthscale,
                  "outputscale": self.outputscale, "rho": self.rho}
        if self._kernel().categorical_dims:
            values["categorical_corr"] = self.categorical_corr
        return values

    def __repr__(self):
        hp = self.as_dict()
        categorical = (f", categorical_corr={hp['categorical_corr']:.6g}"
                       if "categorical_corr" in hp else "")
        return (f"Hyperparameters(varnoise={hp['varnoise']:.6g}, "
                f"lengthscale={hp['lengthscale']}, "
                f"outputscale={hp['outputscale']:.6g}, "
                f"rho={hp['rho']:.6g}{categorical})")


# =============================================================================
# 7. GP fitting, conditioning, and cached MAP predictions
# =============================================================================

class GP:
    """
    GP on X x {0,1} in double precision, parameterized by the cross-group
    correlation rho.

    All interpretable hyperparameters live behind `self.hyperparameters`
    (a live view). Reading it reflects the current model; writing to it edits
    the model in place.
    """

    # ---- construction ----------------------------------------------------
    def __init__(
        self,
        X_train: Union[np.ndarray, torch.Tensor],
        Y_train: Union[np.ndarray, torch.Tensor],
        Map_solution: Optional[torch.Tensor] = None,  # optional MAP solution to initialize the model (for warm start)
        rho: float = 1.0,
        varnoise: float = 1e-2,
        kernel_x: Union[str, gpytorch.kernels.Kernel] = "matern52",
        sigma: Optional[float] = None,                 # initial outputscale sigma_f^2
        lengthscale: Optional[Union[float, Iterable[float], torch.Tensor]] = None,
        inducing_m: int = 1000,
        sparse_threshold: int = 3000,
        learn_inducing_locations: bool = False,
        fix_sigma: bool = False,
        fix_lengthscale: bool = False,
        device: Optional[torch.device] = None,
        *,
        categorical_dims: Iterable[int] = (),
        categorical_corr: float = 0.5,
        Map_coefficients: Optional[torch.Tensor] = None,
    ):
        # Device / dtype -----------------------------------------------------
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float64

        # Data ---------------------------------------------------------------
        self.X_train = _safe_tensor(X_train, device=self.device, dtype=self.dtype, name="X_train")
        self.Y_train = _safe_tensor(Y_train, device=self.device, dtype=self.dtype, name="Y_train").view(-1)
        if self.X_train.dim() != 2 or self.X_train.size(-1) < 2:
            raise ValueError("X_train must be 2-D with at least 2 columns: [spatial..., s].")
        if self.X_train.size(0) != self.Y_train.size(0):
            raise ValueError("X_train and Y_train must have the same number of rows.")

        self.kernel_x = kernel_x
        self.inducing_m = int(inducing_m)
        self.sparse_threshold = int(sparse_threshold)
        self.learn_inducing_locations = bool(learn_inducing_locations)
        self.MAP_solution = None if Map_solution is None else _safe_tensor(
            Map_solution, device=self.device, dtype=self.dtype, name="Map_solution"
        )

        d_spatial = self.X_train.size(-1) - 1  # exclude the class column s

        # Kernel + likelihood ------------------------------------------------
        self.categorical_dims = categorical_columns(d_spatial, categorical_dims)
        self.base_kernel = MixtureKernel(
            kernel_x=self.kernel_x, rho=rho, ard_num_dims=d_spatial,
            categorical_dims=self.categorical_dims, categorical_corr=categorical_corr,
        )
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device).double()

        # Build the model (exact vs sparse) ----------------------------------
        self._build_model()

        # Live hyperparameter view (single source of truth) ------------------
        self.hyperparameters = _LiveHyperparameters(self)

        # Initial hyperparameter values via the live view --------------------
        self.hyperparameters.varnoise = float(varnoise)
        if sigma is not None:
            self.hyperparameters.outputscale = float(sigma)
        if lengthscale is not None:
            self.hyperparameters.lengthscale = lengthscale

        # Optional freezing (explicit, not silent) ---------------------------
        if fix_sigma:
            self.freeze("outputscale")
        if fix_lengthscale:
            self.freeze("lengthscale")

        self._trained = False
        self.rho_fit_results = []
        self.multistart_results = []
        self.Map_coefficients = None
        self._map_cache = None
        self._map_pinv_cache = None
        self.map_pinv_info = None
        if Map_coefficients is not None:
            if self.MAP_solution is None:
                raise ValueError("Map_coefficients requires Map_solution.")
            coefficients = self._map_training_vector(Map_coefficients, "Map_coefficients")
            solution = self._map_training_vector(self.MAP_solution, "MAP_solution")
            # Precomputed coefficients must match the default pseudoinverse.
            with torch.no_grad():
                gram = self.get_gram(self.X_train) if isinstance(kernel_x, gpytorch.kernels.Kernel) else None
                self._cache_map_coefficients(coefficients.clone(), solution, jitter=0.0,
                                             rtol=None, atol=0.0, gram=gram)

    def _build_model(self):
        N = self.X_train.size(0)
        self.is_sparse = N > self.sparse_threshold
        if self.is_sparse:
            Z = self._select_inducing_points()
            self.model = _SVGPWrapper(
                Z, self.base_kernel,
                learn_inducing_locations=self.learn_inducing_locations,
            ).to(self.device).double()
        else:
            self.model = _ExactGPWrapper(
                self.X_train, self.Y_train, self.likelihood, self.base_kernel
            ).to(self.device).double()

    def _select_inducing_points(self) -> torch.Tensor:
        """
        Stratified sampling of inducing points so that BOTH classes are present,
        with valid s in {0, 1} on the last column.
        """
        N = self.X_train.size(0)
        m = min(self.inducing_m, N)
        s = self.X_train[:, -1].long()
        idx0 = torch.nonzero(s == 0, as_tuple=True)[0]
        idx1 = torch.nonzero(s == 1, as_tuple=True)[0]
        if idx0.numel() == 0 or idx1.numel() == 0:
            raise ValueError("Both classes s=0 and s=1 must be present in X_train.")
        m0 = max(1, int(round(m * idx0.numel() / N)))
        m1 = max(1, m - m0)
        sel0 = idx0[torch.randperm(idx0.numel(), device=self.device)[:min(m0, idx0.numel())]]
        sel1 = idx1[torch.randperm(idx1.numel(), device=self.device)[:min(m1, idx1.numel())]]
        return self.X_train[torch.cat([sel0, sel1], dim=0)].clone().to(self.device)

    # ---- data update -----------------------------------------------------
    def Update_data(self, X_new: Union[np.ndarray, torch.Tensor],
                    Y_new: Union[np.ndarray, torch.Tensor]):
        """Replace the training data and rebuild the model (keeps hyperparameters)."""
        self.X_train = _safe_tensor(X_new, device=self.device, dtype=self.dtype, name="X_new")
        self.Y_train = _safe_tensor(Y_new, device=self.device, dtype=self.dtype, name="Y_new").view(-1)
        if self.X_train.size(0) != self.Y_train.size(0):
            raise ValueError("X_new and Y_new must have the same number of rows.")
        hp = self.hyperparameters.as_dict()
        mean_constant = self.model.mean_module.constant.detach().clone()
        self._build_model()
        with torch.no_grad():
            self.model.mean_module.constant.copy_(mean_constant)
        self.hyperparameters.varnoise = hp["varnoise"]
        self.hyperparameters.outputscale = hp["outputscale"]
        self.hyperparameters.lengthscale = hp["lengthscale"]
        self.hyperparameters.rho = hp["rho"]
        self._trained = False
        self.rho_fit_results = []
        self.multistart_results = []
        self.Map_coefficients = None
        self._map_cache = None
        self._map_pinv_cache = None
        self.map_pinv_info = None

    # ---- manual hyperparameter control -----------------------------------
    def set_hyperparameters(self, varnoise=None, lengthscale=None,
                            outputscale=None, rho=None, categorical_corr=None):
        """Convenience setter routing through the live `self.hyperparameters` view."""
        if varnoise is not None:
            self.hyperparameters.varnoise = varnoise
        if lengthscale is not None:
            self.hyperparameters.lengthscale = lengthscale
        if outputscale is not None:
            self.hyperparameters.outputscale = outputscale
        if rho is not None:
            self.hyperparameters.rho = rho
        if categorical_corr is not None:
            self.hyperparameters.categorical_corr = categorical_corr

    def _params_for(self, which: str):
        """Return the raw torch parameters backing a hyperparameter group."""
        k = self.model.covar_module
        if which in ("varnoise", "noise"):
            return [self.likelihood.noise_covar.raw_noise]
        if which in ("lengthscale", "l"):
            raw = getattr(k.spatial, "raw_lengthscale", None)
            return [raw] if raw is not None and raw.numel() else []
        if which in ("outputscale", "sigma"):
            return [k.spatial_scaled.raw_outputscale]
        if which == "rho":
            return [k.raw_rho]
        if which == "categorical_corr":
            return [k.raw_categorical_corr] if k.categorical_dims else []
        raise ValueError(f"Unknown hyperparameter group: {which!r}")

    def freeze(self, which: str):
        """Freeze a hyperparameter group (no gradient during optimisation)."""
        if which == "rho":
            self.model.covar_module.set_learn_rho(False)
            return
        for p in self._params_for(which):
            p.requires_grad_(False)

    def unfreeze(self, which: str):
        """Re-enable optimisation of a hyperparameter group."""
        if which == "rho":
            self.model.covar_module.set_learn_rho(True)
            return
        for p in self._params_for(which):
            p.requires_grad_(True)

    # ---- optimisation ----------------------------------------------------
    def Optimise_hyperparameters(self, fix_varnoise: bool = False, fix_rho: bool = True,
                                 method: str = "MLL", lr: float = 1e-2, iters: int = 200,
                                 batch_size: int = 1024, verbose: bool = True,
                                 *, rho_starts: Optional[Iterable[float]] = None,
                                 n_starts: int = 1, design: str = "sobol",
                                 seed: Optional[int] = None, start_bounds: Optional[dict] = None,
                                 fix_categorical_corr: bool = False):
        """
        Optimise the trainable hyperparameters (exact GP -> marginal log-likelihood,
        sparse GP -> variational ELBO).

        fix_rho=True (default) keeps the rho given at construction: a fairness
        constraint such as rho=1 is never silently optimised away. Pass
        fix_rho=False to estimate rho from the data (it is then constrained to
        |rho| < 1 by the tanh reparameterization).

        n_starts counts the current initialization plus design points for
        trainable lengthscales, variances, mean, rho, and categorical
        correlation. Set design to "sobol" or "lhs" and seed
        to reproduce the design. start_bounds can override lower/upper
        pairs by parameter name. Lengthscales, outputscale and noise variance
        are sampled in log space. Set fix_categorical_corr=True to hold
        the shared categorical correlation fixed.

        Alternatively, pass rho_starts with fix_rho=False to vary only
        the initial rho. Each fit gets a fresh optimizer and the full
        iteration budget. The GP retains the lowest final training loss
        (negative MLL for exact GPs or negative ELBO for sparse GPs).
        The return value is the fitted hyperparameter dictionary.
        Inspect multistart_results for the initial and fitted parameters,
        training loss and selected fit. Explicit rho_starts also populate
        rho_fit_results for compatibility.
        """
        _ = method  # only MLL/ELBO are supported; kept for signature compatibility.

        n_starts = index(n_starts)
        if n_starts < 1:
            raise ValueError("n_starts must be at least one.")
        if rho_starts is not None:
            if fix_rho or n_starts != 1 or start_bounds is not None:
                raise ValueError("Use rho_starts with fix_rho=False and without a generated design.")
            rhos = tuple(float(rho) for rho in rho_starts)
            if not rhos or any(not -1.0 < rho < 1.0 for rho in rhos):
                raise ValueError("rho_starts must contain finite values strictly between -1 and 1.")
            starts = [{"rho": rho} for rho in rhos]
        elif n_starts > 1:
            starts = self._design_starts(
                n_starts - 1, design, seed, start_bounds,
                fix_varnoise, fix_rho, fix_categorical_corr,
            )
        else:
            if start_bounds is not None:
                raise ValueError("start_bounds requires n_starts greater than one.")
            starts = None

        fit_options = (fix_varnoise, fix_rho, fix_categorical_corr, lr, iters, batch_size, verbose)
        if starts is None:
            self._fit_hyperparameters(*fit_options)
            self.multistart_results = []
        else:
            self.multistart_results = self._fit_multistart(starts, *fit_options)
        self.rho_fit_results = [
            dict(rho_start=row["initial"]["rho"], rho_fitted=row["fitted"]["rho"],
                 varnoise=row["fitted"]["varnoise"], objective=row["objective"],
                 loss=row["loss"], selected=row["selected"])
            for row in self.multistart_results
        ] if rho_starts is not None else []
        self._trained = True
        return self.hyperparameters.as_dict()

    def _design_starts(self, count, design, seed, overrides, fix_varnoise, fix_rho, fix_categorical_corr):
        """Build data-scaled bounds for the parameters that will be fitted."""
        y = self.Y_train
        mean = float(y.mean())
        variance = max(float(y.var(unbiased=False)), 1e-6)
        noise_floor = float(self.likelihood.noise_covar.raw_noise_constraint.lower_bound)
        noise_min = max(noise_floor * 1.01, variance * 1e-3, 1e-12)
        bounds = {}
        if not fix_varnoise:
            bounds["varnoise"] = (noise_min, max(variance * 0.5, 10 * noise_min))
        if any(p.requires_grad for p in self._params_for("lengthscale")):
            continuous = [i for i in range(self.X_train.size(1) - 1) if i not in self.categorical_dims]
            features = self.X_train[:, continuous]
            span = (features.amax(0) - features.amin(0)).cpu().numpy()
            span = np.where(span > 0, span, 1.0)
            lengthscale = self.hyperparameters.lengthscale
            if len(span) != len(lengthscale):
                span = np.full_like(lengthscale, np.median(span) if len(span) else 1.0)
            bounds["lengthscale"] = (0.1 * span, 2.0 * span)
        if any(p.requires_grad for p in self._params_for("outputscale")):
            bounds["outputscale"] = (0.1 * variance, 10 * variance)
        if not fix_rho:
            bounds["rho"] = (-0.95, 0.95)
        if self.categorical_dims and not fix_categorical_corr:
            bounds["categorical_corr"] = (0.05, 0.95)
        if self.model.mean_module.raw_constant.requires_grad:
            bounds["mean"] = (mean - np.sqrt(variance), mean + np.sqrt(variance))

        for name, (lo, hi) in (overrides or {}).items():
            if name not in bounds:
                raise ValueError(f"Cannot set start bounds for unknown or fixed parameter {name!r}.")
            shape = np.shape(bounds[name][0])
            bounds[name] = (np.broadcast_to(lo, shape), np.broadcast_to(hi, shape))
        if "rho" in bounds and not -1 < bounds["rho"][0] < bounds["rho"][1] < 1:
            raise ValueError("rho start bounds must lie strictly between -1 and 1.")
        if "categorical_corr" in bounds and not 0 < bounds["categorical_corr"][0] < bounds["categorical_corr"][1] < 1:
            raise ValueError("categorical_corr start bounds must lie strictly between 0 and 1.")
        if "varnoise" in bounds and bounds["varnoise"][0] <= noise_floor:
            raise ValueError("varnoise start bounds must exceed the likelihood noise floor.")
        return [{}] + sample_hyperparameter_starts(bounds, count, design=design, seed=seed)

    def _fit_hyperparameters(self, fix_varnoise, fix_rho, fix_categorical_corr, lr, iters, batch_size, verbose):
        if fix_varnoise:
            self.freeze("varnoise")
        else:
            self.unfreeze("varnoise")
        self.freeze("rho") if fix_rho else self.unfreeze("rho")
        self.freeze("categorical_corr") if fix_categorical_corr else self.unfreeze("categorical_corr")

        self.model.train()
        self.likelihood.train()

        trainable = [p for p in set(self.model.parameters()) | set(self.likelihood.parameters())
                     if p.requires_grad]
        if not trainable:
            if verbose:
                print("[Optimise] Nothing to optimise: all hyperparameters are frozen.")
            return
        optimizer = torch.optim.Adam(trainable, lr=lr)

        if self.is_sparse:
            self._fit_sparse(optimizer, iters=iters, batch_size=batch_size, verbose=verbose)
        else:
            self._fit_exact(optimizer, iters=iters, verbose=verbose)

        self.model.eval()
        self.likelihood.eval()

    @torch.no_grad()
    def _training_loss(self, batch_size):
        """Evaluate the final training objective per observation."""
        xs, targets = self.X_train, self.Y_train
        if not self.is_sparse:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            # forward supplies the prior at the training inputs, even in eval mode.
            with gpytorch.settings.fast_computations(log_prob=False):
                return float(-mll(self.model.forward(xs), targets))

        elbo = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(xs))
        total = 0.0
        for x_batch, y_batch in zip(xs.split(batch_size), targets.split(batch_size)):
            total -= len(x_batch) * float(elbo(self.model(x_batch), y_batch))
        return total / len(xs)

    def _fit_multistart(self, starts, fix_varnoise, fix_rho, fix_categorical_corr,
                        lr, iters, batch_size, verbose):
        kernel = self.model.covar_module
        noise = self.likelihood.noise_covar.raw_noise
        initial_state = deepcopy(self.model.state_dict()), deepcopy(self.likelihood.state_dict())
        initial_modes = self.model.training, self.likelihood.training
        initial_learn_rho = kernel.learn_rho
        configured = [kernel.raw_rho, noise, *self._params_for("categorical_corr")]
        initial_grad_flags = [parameter.requires_grad for parameter in configured]
        cuda_devices = [self.X_train.device] if self.X_train.is_cuda else []
        objective = "negative ELBO" if self.is_sparse else "negative MLL"

        def load_state(state):
            self.model.load_state_dict(state[0])
            self.likelihood.load_state_dict(state[1])
            self.model.train()  # Clear caches associated with the previous fit.

        def parameters():
            values = self.hyperparameters.as_dict()
            values["lengthscale"] = values["lengthscale"].tolist()
            values["mean"] = float(self.model.mean_module.constant.detach())
            return values

        results = []
        best_state, best_index, best_loss = None, None, float("inf")
        try:
            for start_index, start in enumerate(starts):
                load_state(initial_state)
                kernel.learn_rho = initial_learn_rho
                self.set_hyperparameters(**{key: value for key, value in start.items() if key != "mean"})
                if "mean" in start:
                    self.model.mean_module.initialize(raw_constant=start["mean"])
                self.freeze("rho") if fix_rho else self.unfreeze("rho")
                initial = parameters()
                if verbose:
                    print(f"[Multistart] Fit {start_index + 1}/{len(starts)}")

                try:
                    # Replay variational initialization and minibatch ordering.
                    with torch.random.fork_rng(devices=cuda_devices):
                        self._fit_hyperparameters(
                            fix_varnoise, fix_rho, fix_categorical_corr, lr, iters, batch_size, verbose
                        )
                        loss = self._training_loss(batch_size)
                except (NanError, NotPSDError, torch.linalg.LinAlgError) as error:
                    if verbose:
                        print(f"[Multistart] Fit failed: {error}")
                    loss = float("inf")
                if not np.isfinite(loss):
                    loss = float("inf")
                if loss < best_loss:
                    best_loss = loss
                    best_index = len(results)
                    best_state = deepcopy(self.model.state_dict()), deepcopy(self.likelihood.state_dict())

                results.append(dict(
                    start_index=start_index, initial=initial, fitted=parameters(),
                    objective=objective, loss=loss, selected=False,
                ))
                if verbose:
                    print(f"[Multistart] rho={self.hyperparameters.rho:.6g}, {objective}={loss:.6g}")

            if best_state is None:
                raise RuntimeError("Every start failed to produce a finite training loss.")
            load_state(best_state)
            self.model.eval()
            self.likelihood.eval()
        except Exception:
            load_state(initial_state)
            kernel.learn_rho = initial_learn_rho
            for parameter, requires_grad in zip(configured, initial_grad_flags):
                parameter.requires_grad_(requires_grad)
            self.model.train(initial_modes[0])
            self.likelihood.train(initial_modes[1])
            raise

        results[best_index]["selected"] = True
        if verbose:
            print(f"[Multistart] Selected fit {best_index + 1}")
        return results

    def _fit_exact(self, optimizer, iters: int, verbose: bool):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        step = max(1, iters // 10)
        for i in range(iters):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.Y_train)
            loss.backward()
            optimizer.step()
            if verbose and ((i + 1) % step == 0 or i == iters - 1):
                print(f"[ExactGP] Iter {i + 1}/{iters} - Loss: {loss.item():.6f}")

    def _fit_sparse(self, optimizer, iters: int, batch_size: int, verbose: bool):
        num_data = self.X_train.size(0)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=num_data)
        step = max(1, iters // 10)
        for epoch in range(iters):
            # Manual mini-batching over data already living on `self.device`
            # (avoids DataLoader/pin_memory/generator device pitfalls).
            perm = torch.randperm(num_data, device=self.device)
            epoch_loss = 0.0
            for start in range(0, num_data, batch_size):
                b = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = -mll(self.model(self.X_train[b]), self.Y_train[b])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            if verbose and ((epoch + 1) % step == 0 or epoch == iters - 1):
                print(f"[SVGP] Epoch {epoch + 1}/{iters} - Loss: {epoch_loss:.6f}")

    # ---- reporting -------------------------------------------------------
    def print_hyperparameters(self):
        if not hasattr(self, "model"):
            print("Model not initialized.")
            return
        hp = self.hyperparameters.as_dict()
        k = self.model.covar_module
        status = "estimated" if k.learn_rho else "fixed"
        print("\n=== GP Hyperparameters ===")
        print(f"Observation noise (sigma_n^2): {hp['varnoise']:.6f}")
        print(f"Output scale (sigma_f^2):      {hp['outputscale']:.6f}")
        print(f"Lengthscale:                   {hp['lengthscale']}")
        print(f"Cross-group correlation rho:   {hp['rho']:.6f}  ({status})")
        if "categorical_corr" in hp:
            print(f"Categorical correlation:      {hp['categorical_corr']:.6f}")
        if k.is_collapsed:
            print("  -> rho = 1: the sensitive label has no direct effect")
        print("==========================\n")

    # ---- tensor plumbing -------------------------------------------------
    def _ensure_tensor(self, X, as_long: bool = False, name: str = "X"):
        return _safe_tensor(X, device=self.device, dtype=self.dtype, as_long=as_long, name=name)

    # ---- gram / mean utilities ------------------------------------------
    def get_gram(self, X, Y=None) -> torch.Tensor:
        Xt = self._ensure_tensor(X, name="get_gram.X")
        if Y is None:
            K_linop = self.model.covar_module(Xt)
        else:
            Yt = self._ensure_tensor(Y, name="get_gram.Y")
            K_linop = self.model.covar_module(Xt, Yt)
        try:
            return K_linop.to_dense()
        except Exception:
            return K_linop.evaluate().to_dense()

    def get_mean(self, X) -> torch.Tensor:
        Xt = self._ensure_tensor(X, name="get_mean.X")
        with torch.no_grad():
            m = self.model.mean_module(Xt)
        return m.detach()

    def _posterior_on(self, X):
        Xt = self._ensure_tensor(X, name="_posterior_on.X")
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.likelihood(self.model(Xt))
        return posterior

    def get_cond_mean(self, X) -> torch.Tensor:
        return self._posterior_on(X).mean.detach()

    def _map_training_vector(self, values, name):
        values = self._ensure_tensor(values, name=name).detach()
        if values.shape not in ((len(self.X_train),), (len(self.X_train), 1)):
            raise ValueError(f"{name} must contain one value per training row.")
        return values.reshape(-1, 1)

    def _map_kernel_state(self):
        kernel = self.model.covar_module
        return (id(kernel), kernel.learn_rho, kernel.categorical_dims, kernel.training,
                getattr(kernel.spatial, "nu", None), torch.get_default_dtype())

    def _map_cache_inputs(self, solution=None, inputs=None):
        kernel = self.model.covar_module
        values = (self.X_train if inputs is None else inputs, *kernel.parameters(), *kernel.buffers())
        return values if solution is None else (*values, solution)

    @staticmethod
    def _map_same_inputs(current, previous):
        return len(current) == len(previous) and all(
            a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
            for a, b in zip(current, previous)
        )

    def _cache_map_coefficients(self, coefficients, solution, jitter, rtol, atol, gram=None):
        # Compare values rather than tensor versions alone: GPyTorch setters
        # can update a parameter through .data without bumping its version.
        inputs = tuple(value.detach().clone() for value in self._map_cache_inputs(solution))
        info = None if self.map_pinv_info is None else dict(self.map_pinv_info)
        self._map_cache = ((jitter, rtol, atol), self._map_kernel_state(), inputs, gram, info)
        self.Map_coefficients = coefficients.detach()

    @torch.no_grad()
    def _map_pinv_factors(self, inputs, jitter, rtol, atol):
        custom = isinstance(self.kernel_x, gpytorch.kernels.Kernel)
        gram = self.get_gram(inputs) if custom else None
        cache = getattr(self, "_map_pinv_cache", None)
        state = self._map_kernel_state()
        current = self._map_cache_inputs(inputs=inputs)
        if (cache is not None and cache["settings"] == (jitter, rtol, atol)
                and cache["state"] == state and self._map_same_inputs(current, cache["inputs"])
                and (not custom or torch.equal(gram, cache["gram"]))):
            return cache["factors"]
        if gram is None:
            gram = self.get_gram(inputs)
        factors = symmetric_pinv_factors(gram, jitter=jitter, rtol=rtol, atol=atol)
        self._map_pinv_cache = dict(
            settings=(jitter, rtol, atol), state=state,
            inputs=tuple(value.detach().clone() for value in current),
            gram=gram.detach().clone() if custom else None, factors=factors,
        )
        return factors

    @torch.no_grad()
    def get_pinv_gram(self, X=None, *, jitter=0.0, rtol=None, atol=0.0) -> torch.Tensor:
        """Return the prior training Gram pseudoinverse, or that at X.

        The symmetric eigendecomposition is cached. With default tolerances,
        the cutoff is n * machine epsilon * the largest eigenvalue magnitude.
        Positive jitter explicitly changes the matrix being pseudoinverted.
        """
        inputs = self.X_train if X is None else self._ensure_tensor(X, name="get_pinv_gram.X")
        jitter, rtol, atol = _pseudoinverse_settings(jitter, rtol, atol)
        vectors, reciprocals, info = self._map_pinv_factors(inputs, jitter, rtol, atol)
        inverse = (vectors * reciprocals) @ vectors.T
        self.map_pinv_info = dict(info)
        return (inverse + inverse.T) * 0.5

    @torch.no_grad()
    def compute_map_coefficients(self, values=None, *, jitter=0.0, rtol=None, atol=0.0):
        """Return K^+ times one vector (N,) or multiple columns (N, L).

        This shares one eigendecomposition across solutions and predictions.
        It does not assign MAP_solution or Map_coefficients.
        """
        if values is None:
            values = self.MAP_solution
        if values is None:
            raise ValueError("MAP solution not available. Solve the fairness problem first.")
        targets = self._ensure_tensor(values, name="compute_map_coefficients.values").detach()
        if targets.ndim == 1:
            targets = targets[:, None]
        if targets.ndim != 2 or targets.size(0) != len(self.X_train) or targets.size(1) == 0:
            raise ValueError("values must have shape (N,) or (N, L) with one row per training input.")
        jitter, rtol, atol = _pseudoinverse_settings(jitter, rtol, atol)
        vectors, reciprocals, info = self._map_pinv_factors(self.X_train, jitter, rtol, atol)
        coefficients = apply_symmetric_pinv(vectors, reciprocals, targets)
        discarded = vectors[:, reciprocals == 0].T @ targets
        self.map_pinv_info = dict(info, discarded_component_norm=float(torch.linalg.vector_norm(discarded)))
        return coefficients

    @torch.no_grad()
    def MAP_pred(self, X, jitter: float = 0.0, *, rtol=None, atol=0.0) -> torch.Tensor:
        """Predict with K(X, Z) K(Z, Z)^+ MAP_solution and cached coefficients.

        Constructor-supplied Map_coefficients must match Map_solution and
        the initial training pseudoinverse for the default tolerances and
        jitter=0. Changes to the solution, kernel, jitter or tolerances
        invalidate coefficients. The kernel decomposition is shared across
        MAP solutions. Singular systems use minimum-norm least-squares
        coefficients and project incompatible training values.
        """
        if self.MAP_solution is None:
            raise ValueError("MAP solution not available. Solve the fairness problem first.")
        Xt = self._ensure_tensor(X)
        solution = self._map_training_vector(self.MAP_solution, "MAP_solution")
        jitter, rtol, atol = _pseudoinverse_settings(jitter, rtol, atol)
        coefficients = getattr(self, "Map_coefficients", None)
        cache = getattr(self, "_map_cache", None)
        # Custom kernels can depend on attributes outside their tensor state.
        custom_kernel = isinstance(self.kernel_x, gpytorch.kernels.Kernel)
        gram = self.get_gram(self.X_train) if custom_kernel else None
        if coefficients is not None and cache is not None and cache[:2] == ((jitter, rtol, atol), self._map_kernel_state()):
            current, previous = self._map_cache_inputs(solution), cache[2]
            same_inputs = self._map_same_inputs(current, previous)
            same_gram = not custom_kernel or (cache[3] is not None and torch.equal(gram, cache[3]))
            if same_inputs and same_gram:
                self.map_pinv_info = None if cache[4] is None else dict(cache[4])
                return (self.get_gram(Xt, self.X_train) @ coefficients).squeeze(-1)
        coefficients = self.compute_map_coefficients(solution, jitter=jitter, rtol=rtol, atol=atol)
        self._cache_map_coefficients(coefficients, solution, jitter, rtol, atol, gram)
        prediction = (self.get_gram(Xt, self.X_train) @ coefficients).squeeze(-1)
        return prediction


    def get_cond_gram(self, X, Y=None) -> torch.Tensor:
        Xt = self._ensure_tensor(X, name="get_cond_gram.X")
        if Y is None:
            return self._posterior_on(Xt).covariance_matrix.detach()
        Yt = self._ensure_tensor(Y, name="get_cond_gram.Y")
        XY = torch.cat([Xt, Yt], dim=0)
        cov = self._posterior_on(XY).covariance_matrix.detach()
        m = Xt.size(0)
        return cov[:m, m:]

    def get_inv_cond_gram(self, X) -> torch.Tensor:
        cov = self.get_cond_gram(X)
        cov = cov + 1e-8 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        L = torch.linalg.cholesky(cov)
        inv_L = torch.linalg.solve_triangular(
            L, torch.eye(L.size(0), device=L.device, dtype=L.dtype), upper=False
        )
        return inv_L.T @ inv_L

    # ---- sampling --------------------------------------------------------
    def Sample(self, n: int, X=None) -> torch.Tensor:
        """Draw n prior samples of the latent function at X (defaults to X_train)."""
        if X is None:
            X = self.X_train
        Xt = self._ensure_tensor(X, name="Sample.X")
        with torch.no_grad():
            mean = self.model.mean_module(Xt)
            cov = self.model.covar_module(Xt).to_dense()
            cov = cov + 1e-8 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            L = torch.linalg.cholesky(cov)
            z = torch.randn(n, cov.size(0), device=cov.device, dtype=cov.dtype)
            samples = mean.unsqueeze(0) + z @ L.T
        return samples.detach()

    def Sample_cond(self, n: int, X=None) -> torch.Tensor:
        """Draw n posterior samples at X (defaults to X_train)."""
        if X is None:
            X = self.X_train
        Xt = self._ensure_tensor(X, name="Sample_cond.X")
        post = self._posterior_on(Xt)
        mean = post.mean
        cov = post.covariance_matrix
        cov = cov + 1e-8 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        L = torch.linalg.cholesky(cov)
        z = torch.randn(n, cov.size(0), device=cov.device, dtype=cov.dtype)
        samples = mean.unsqueeze(0) + z @ L.T
        return samples.detach()


# =============================================================================
# 8. Batched MAP predictions along a fairness-penalty path
# =============================================================================

@torch.no_grad()
def predict_map_path(gp, xs, solutions, *, jitter=0.0, rtol=None, atol=0.0):
    """Return observed, group-0, and group-1 predictions of shape (L, N).

    The last input column is the binary sensitive label. Each of the L
    solutions contains one latent value per training input. The same
    prior Gram pseudoinverse is applied to all solutions in one call to
    GP.compute_map_coefficients. The GP caches the spectral decomposition
    across paths and evaluation inputs. Eigenvalues are retained according
    to rtol and atol, as in GP.MAP_pred.
    """
    train = gp.X_train
    inputs = torch.as_tensor(xs, dtype=train.dtype, device=train.device)
    if inputs.ndim != 2 or inputs.shape[1] != train.shape[1]:
        raise ValueError("Prediction inputs must have the same columns as the training inputs.")
    labels = inputs[:, -1]
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("The last prediction input column must contain binary group labels.")

    targets = [torch.as_tensor(z, dtype=train.dtype, device=train.device).reshape(-1)
               for z in solutions]
    if not targets or any(z.numel() != len(train) for z in targets):
        raise ValueError("Supply one or more solutions with one value per training input.")

    weights = gp.compute_map_coefficients(
        torch.stack(targets, dim=1), jitter=jitter, rtol=rtol, atol=atol
    )
    group_predictions = []
    for label in (0, 1):
        counterfactual = inputs.clone()
        counterfactual[:, -1] = label
        group_predictions.append((gp.get_gram(counterfactual, train) @ weights).T)
    pred0, pred1 = group_predictions
    observed = torch.where(labels == 0, pred0, pred1)
    return observed, pred0, pred1
