import torch
torch.set_default_dtype(torch.float64)

# =========================
# Utility
# =========================
def _to_device_like(value, reference_tensor):
    if torch.is_tensor(value):
        return value.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    else:
        return torch.tensor(value, device=reference_tensor.device, dtype=reference_tensor.dtype)
# =========================
# Sum of kernels
# =========================
class SumKernel:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def get_gram(self, X, Y):
        return self.k1.get_gram(X, Y) + self.k2.get_gram(X, Y)

    def get_gram_diff(self, X, Y):
        return self.k1.get_gram_diff(X, Y) + self.k2.get_gram_diff(X, Y)

# =========================
# Base Kernel
# =========================
class Kernel:
    def __init__(self, lengthscale=1.0, scale=1.0):
        self.lengthscale = lengthscale
        self.scale = scale
        
    def __add__(self, other):
        return SumKernel(self, other)
    
    def get_gram(self, X, Y):
        raise NotImplementedError

    def get_gram_diff(self, X, Y):
        raise NotImplementedError

# =========================
# Gaussian / RBF kernel
# =========================
class GaussianKernel(Kernel):
    def get_gram(self, X, Y):
        ell = _to_device_like(self.lengthscale, X)
        s   = _to_device_like(self.scale, X)
        diff = X - Y.T  # X, Y are (n,1)
        d2 = diff**2
        return s * torch.exp(-0.5 * d2 / ell**2)

    def get_gram_diff(self, X, Y):
        ell = _to_device_like(self.lengthscale, X)
        s   = _to_device_like(self.scale, X)
        diff = X - Y.T
        K = s * torch.exp(-0.5 * diff**2 / ell**2)
        return K * (-diff / ell**2)

# =========================
# Matérn kernel (nu=0.5,1.5,2.5)
# =========================
class MaternKernel(Kernel):
    def __init__(self, nu=1.5, lengthscale=1.0, scale=1.0):
        assert nu in (0.5, 1.5, 2.5)
        super().__init__(lengthscale, scale)
        self.nu = nu

    def get_gram(self, X, Y):
        ell = _to_device_like(self.lengthscale, X)
        s   = _to_device_like(self.scale, X)
        diff = X - Y.T
        r = torch.abs(diff) + 1e-12  # avoid division by zero

        if self.nu == 0.5:
            K = s * torch.exp(-r / ell)
        elif self.nu == 1.5:
            sqrt3 = torch.sqrt(torch.tensor(3.0, device=X.device, dtype=X.dtype))
            r_scaled = sqrt3 * r / ell
            K = s * (1 + r_scaled) * torch.exp(-r_scaled)
        else:  # nu=2.5
            sqrt5 = torch.sqrt(torch.tensor(5.0, device=X.device, dtype=X.dtype))
            r_scaled = sqrt5 * r / ell
            K = s * (1 + r_scaled + r_scaled**2 / 3) * torch.exp(-r_scaled)
        return K

    def get_gram_diff(self, X, Y):
            ell = _to_device_like(self.lengthscale, X)
            s   = _to_device_like(self.scale, X)
            diff = X - Y.T
            r = torch.abs(diff)
            sign = torch.sign(diff)

            # General formula for every nu
            if self.nu == 0.5:
                K = s * torch.exp(-r / ell)
                dK_dr = -K / ell
            elif self.nu == 1.5:
                sqrt3 = torch.sqrt(torch.tensor(3.0, device=X.device, dtype=X.dtype))
                r_scaled = sqrt3 * r / ell
                dK_dr = s * ( -3*r/(ell**2) ) * torch.exp(-r_scaled)  # simplified
            else:  # nu=2.5
                sqrt5 = torch.sqrt(torch.tensor(5.0, device=X.device, dtype=X.dtype))
                r_scaled = sqrt5 * r / ell
                dK_dr = s * ( -5*r/(3*ell**2) *(1 + r_scaled)) * torch.exp(-r_scaled)  # simplified
            return dK_dr * sign



# ===========================================
# Median heuristic
# ===========================================

def median_heuristic(X: torch.Tensor) -> torch.Tensor:
    Dist = torch.cdist(X, X, p=2.0)**2
    lower_tri = torch.tril(Dist, diagonal=-1).reshape(-1)
    lower_tri = lower_tri[lower_tri != 0]
    return (lower_tri.median() / 2).sqrt()

