"""
opt_method.py

Solve by gradient descent:
    min_{X, Y}  (v - mu)^T Gamma (v - mu)  +  lambda*D(empirical(X), empirical(Y))

- Where D is either MMD or Sinkhorn divergence.
- Uses PyTorch for autodiff and optimization.
- Uses GeomLoss.SamplesLoss to compute a differentiable Sinkhorn cost.
- Uses typing for Tuple and List
"""

import torch
from geomloss import SamplesLoss  # To get the Sinkhorn divergence 
from scipy.stats import ks_2samp

# Use double precision everywhere
torch.set_default_dtype(torch.float64)
torch.set_default_device("cpu") #torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
#  Quadratic function 
# ====================

def quadratic_term(X: torch.Tensor, Y: torch.Tensor, mu: torch.Tensor, Gamma: torch.Tensor) -> torch.Tensor:
    """
    Compute quadratic form (v-mu)^T Gamma (v-mu) where v = concat(vec(X), vec(Y))

    Inputs:
        X : torch.Tensor, shape (n_x, d_x)
        Y : torch.Tensor, shape (n_y, d_y)
        mu : torch.Tensor, shape (N_tot,)
            target vector of same length as v (flattened)
        Gamma : torch.Tensor, shape (N_tot, N_tot)
            symmetric positive-definite matrix

    Returns:
        scalar torch.Tensor
    """
    v = torch.cat([X.reshape(-1), Y.reshape(-1)])  # (N_total,)
    diff = v - mu.reshape(-1)
    # compute scalar = diff^T (Gamma diff)
    return (diff @ (Gamma @ diff)).reshape(())


# ============================================================
#  Unbiased MMD² and its analytic gradient
# ============================================================

def mmd_loss(X: torch.Tensor, Y: torch.Tensor, kernel: object) -> torch.Tensor:
    """
    Compute the MMD between P_X and P_Y:
        MMD^2 =
        (1/n^2) 1_n^T k(X,X) 1_n - (2/(n m)) 1_n^T k(X,Y) 1_m + (1/m^2) 1_m^T k(Y,Y)
    X: (n,1)
    Y: (m,1)
    kernel: object with method get_gram(A,B) -> (n,m)
    """
    n, m = X.shape[0], Y.shape[0]

    # Compute Gram matrices and ones vectors
    K_xx = kernel.get_gram(X, X)    # (n,n)
    K_xy = kernel.get_gram(X, Y)    # (n,m)
    K_yy = kernel.get_gram(Y, Y)    # (m,m)

    one_n = torch.ones(n, 1, device=X.device, dtype=X.dtype)
    one_m = torch.ones(m, 1, device=Y.device, dtype=Y.dtype)

    # Computation of the 3 terms in MMD^2
    term_xx = (one_n.T @ K_xx @ one_n) / (n**2)
    term_xy = (one_n.T @ K_xy @ one_m) / (n*m)
    term_yy = (one_m.T @ K_yy @ one_m) / (m**2)

    return (term_xx - 2*term_xy + term_yy).squeeze()


def mmd_grad(X: torch.Tensor, Y: torch.Tensor, kernel: object):
    """
    Compute the gradient of the biased MMD:
        grad = 2 * [[ k1(X,X),   -k1(X,Y) ],
                    [ -k1(Y,X),  k1(Y,Y) ]]  [1_n ; 1_m]
    X : (n,1)
    Y : (m,1)
    returns (grad_X : (n,1), grad_Y : (m,1))
    """
    n, m = X.shape[0], Y.shape[0]

    # Ones vectors
    one_n = torch.ones(n, 1, device=X.device, dtype=X.dtype)
    one_m = torch.ones(m, 1, device=Y.device, dtype=Y.dtype)

    # k1(X,X) shape: (n,n,1)
    k1_xx = kernel.get_gram_diff(X, X)     # (n,n)
    k1_xy = kernel.get_gram_diff(X, Y)     # (n,m)
    k1_yx = kernel.get_gram_diff(Y, X)     # (m,n)
    k1_yy = kernel.get_gram_diff(Y, Y)     # (m,m)

    # Block-matrix-vector product
    grad_X = 1/(n*m) * ((m/n)*k1_xx @ one_n - k1_xy @ one_m)       # (n,1)
    grad_Y = 1/(n*m) * (-k1_yx @ one_n + (n/m)*k1_yy @ one_m)      # (m,1)

    return grad_X, grad_Y

# =========================================================
#  Optimizations Methodes of quadratic for MMD or Sinkhorn
# =========================================================

### Naive gradient descent with Adam optimizer

def Optimize_fair_quad(X0: torch.Tensor, mu: torch.Tensor, Gamma: torch.Tensor, loss_type='MMD', 
                       autograd=True, blur=0.05, p=2, lr=1e-2,n_iter=500, lambd=1.0, verbose=True,
                       X1=None, kernel=None, device=None, S=None, tol=1e-6):
    """
    Optimize X and Y via gradient-based optimization minimizing
        Q(X,Y) = quadratic_term(X,Y) + lambd * D(X, Y)
    where D is either MMD or Sinkhorn divergence.

    Inputs:
        X0 : torch.Tensor (n_x, 1)
        X1 : torch.Tensor (n_y, 1)
        mu : torch.Tensor, shape (N_total,) where N_total = n_x*1 + n_y*1
        Gamma : torch.Tensor, shape (N_total, N_total), symmetric PD
        loss_type : 'MMD' or 'Sinkhorn'
        kernel : kernel type for MMD ('gaussian', 'matern32', 'matern52')
        blur : double, entropic regularization scale for Sinkhorn
        p : int, ground cost exponent for Sinkhorn
        lr : double, learning rate
        n_iter : int, number of optimization iterations
        lambd : double, weight of the divergence term
        verbose : bool, whether to print progress
        device : torch.device or None
        S : Sensitive attribute vector (optional) defining groups in X0 and Y0

    Returns:
        X_opt, Y_opt : optimized tensors (requires_grad=False)
        logs : dict with history of 'loss', 'quad', 'D'
    """

    # Device
    device = torch.device(device) if device is not None else mu.device
    
    # The case with sensitive attribute S
    get_S = S is not None
    if get_S:
        ind0, ind1 = torch.nonzero(S == 0, as_tuple=True)[0], torch.nonzero(S == 1, as_tuple=True)[0]
        perm = torch.cat([ind0, ind1], dim=0)
        Gamma = Gamma[perm][:,perm]
        mu = mu[perm]
        X,Y = X0[ind0].clone().detach().double(), X0[ind1].clone().detach().double()
    else:    
        X,Y = X0.clone().detach().double(), X1.clone().double().to(device).double()

    mu = mu.double()
    Gamma = Gamma.double()

    # Convert to parameters we can optimize
    
    Xp = torch.nn.Parameter(X)
    Yp = torch.nn.Parameter(Y)

    optimizer = torch.optim.Adam([Xp, Yp], lr=lr)

    if loss_type == 'Sinkhorn':
        sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, debias=True)
        def D_func(A, B):
            return sinkhorn(A, B)
        
    elif loss_type == 'MMD':
        if kernel is None:
            raise ValueError("For MMD loss, provide a kernel from Fair_GPs.Kernel_classe.")
        def D_func(A, B):
            return mmd_loss(A, B, kernel=kernel)
    
    logs = {"loss": [], "quad": [], "dist": [], loss_type: []}
    
    # ---- Early stopping variables ----
    prev_loss = None

    # ---- Optimization loop ----
    for it in range(1, n_iter + 1):
        optimizer.zero_grad()

        quad = quadratic_term(Xp, Yp, mu, Gamma)
        dist = D_func(Xp, Yp)
        loss = quad + lambd * dist
        
        if autograd:
            loss.backward() # Backprop and step
        
        elif loss_type == 'MMD': # Analytic computation of gradients
            v = torch.cat((Xp, Yp),dim=0)
            diff = (v - mu)

            # ---- Analytic gradients ----
            grad_quad = 2.0 * Gamma.matmul(diff)
            grad_quad_X, grad_quad_Y = grad_quad[:Xp.shape[0]],grad_quad[Xp.shape[0]:]
            grad_mmd_X, grad_mmd_Y = mmd_grad(Xp, Yp, kernel)

            # Total gradients for Adam
            Xp.grad = grad_quad_X + lambd * grad_mmd_X
            Yp.grad = grad_quad_Y + lambd * grad_mmd_Y
        
        else:
            raise ValueError("Autograd is only supported for MMD loss.")
        
        optimizer.step()
        curr_loss = loss.item()
        
        logs["loss"].append(curr_loss)
        logs["quad"].append(quad.item())
        logs["dist"].append(dist.detach())
        logs[loss_type].append(dist.item())
        

        if (it % max(1, n_iter // 10) == 0):
            if verbose:
                ks = ks_2samp(Xp.detach().cpu().numpy().reshape(-1),Yp.detach().cpu().numpy().reshape(-1))
                print(f"Iter {it:4d} | Loss {loss.item():.6f} | Quad {quad.item():.6f} | D {dist.item():.6f} | KS {ks.statistic:.6f}")
            if prev_loss is not None:
                rel_change = abs(curr_loss - prev_loss) / max(1.0, abs(prev_loss))
                if rel_change < tol:
                    if verbose:
                        print(f"Early stopping at iter {it} | rel_change={rel_change:.2e}")
                    break
            prev_loss = curr_loss

    X_opt = Xp.detach()
    Y_opt = Yp.detach()

    
    torch.cuda.empty_cache()
    if get_S: #With sensitive attribute S
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(len(perm))
        return torch.cat([X_opt, Y_opt], dim=0)[inv_perm], D_func(X_opt, Y_opt), logs
    
    return X_opt, Y_opt, D_func(X_opt, Y_opt), logs


# Multi-start optimization method

def Optimize__multistart(X0_list: list, mu: torch.Tensor, Gamma: torch.Tensor, Lr_list: list,
                         loss_type='MMD', autograd=True, blur=0.05, p=2,n_iter=500, lambd=1.0,
                         verbose=True, X1_list=None, kernel=None, device=None, S=None, tol=1e-6):
    """
    Multi-start optimization wrapper for Optimize_fair_quad.
    Inputs:
        X0_list : list of torch.Tensor (n_x, 1)
        X1_list : list of torch.Tensor (n_y, 1)
        mu : torch.Tensor, shape (N_total,) where N_total = n_x*1 + n_y*1
        Gamma : torch.Tensor, shape (N_total, N_total), symmetric PD
        loss_type : 'MMD' or 'Sinkhorn'
        kernel : kernel type for MMD ('gaussian', 'matern32', 'matern52')
        blur : double, entropic regularization scale for Sinkhorn
        p : int, ground cost exponent for Sinkhorn
        lr : double, learning rate
        n_iter : int, number of optimization iterations
        lambd : double, weight of the divergence term
        verbose : bool, whether to print progress
        device : torch.device or None
        S : Sensitive attribute vector (optional) defining groups in X0 and Y0

    Returns:
        X_opt, Y_opt : optimized tensors (requires_grad=False)
        logs : dict with history of 'loss', 'quad', 'D'
    """

    best_loss = float("inf")
    best_result = None
    final_loss = None

    K = len(X0_list)

    for k in range(K):
        if verbose:
            print(f"= Multi-start {k+1}/{K} =")

        X0 = X0_list[k]
        X1 = X1_list[k] if X1_list is not None else None

        result = Optimize_fair_quad(X0=X0,X1=X1,mu=mu,
            Gamma=Gamma,
            loss_type=loss_type,
            autograd=autograd,
            blur=blur,
            p=p,
            lr=Lr_list[k],
            n_iter=n_iter,
            lambd=lambd,
            verbose=verbose,
            kernel=kernel,
            device=device,
            S=S,
            tol=tol
        )
        logs = result[-1]
        final_loss = logs["loss"][-1]

        if final_loss is not None and final_loss < best_loss:
            best_loss = final_loss
            best_result = result

    return best_result


