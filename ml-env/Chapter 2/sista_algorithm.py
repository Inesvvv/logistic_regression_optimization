"""
Created 21/02/2026
This file implements the SISTA algorithm with proximal gradient descent 
Later need to implement it with newton method and compare convergence

Here we can't use gradient descent because the L1 norm is not differentiable at 0 so we use ISTA (proximal gradient descent). 
SISTA alternates between: 
1. Sinkhorn step --> updating u and v 
2. ISTA step --> updating beta to move β in the direction that makes π closer to ^π, while applying an L1 sparsity penalty (soft-thresholding)
Code thus has two core parts: 
sinkhorn_one_pass(...) and beta update via gradient + soft_threshold

24/02/2026 
Add sista with newton method instead of proximal gradient descent. Components are mostly the same, step 5 is the only thing that changes. 
"""

import numpy as np

"""
Sinkhorn uses a lot of log sum of exp formulas so we first implement a utility function
Issue is that exp can overflow when inputs are too large so trick is that we substract the max of a from each explonential in the sum
"""
def logsumexp(a, axis=None, keepdims = False):
    a_max = np.max(a, axis=None, keepdims = True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=None, keepdims= True))
    if not keepdims: 
        out = np.squeeze(out, axis=axis)
    return out 

"""
Soft thresholding implementation --> its the proximal operator for lambda l1 norm, its what creates sparsity
"""
def soft_threshold(z, lam):
    return np.sign(z)*np.max(np.abs(z)-lam, 0.0)

"""
Cost, output is a NxN matrix 
"""
def build_cost(beta, D):
    return np.tensordot(beta,D, axes=(0,0))

"""
Sinkhorn to update u and v
"""
def sinkhorn_one_pass(p,q,C,u,v):
    u = np.log(p) - logsumexp(v[None,:] - C, axis=1)
    v = np.log(p) - logsumexp(u[:, None] - C, axis=0)

"""
compute pi to reconstruct the current transport plan 
"""
def compute_pi(u,v,C):
    return np.exp(u[:, None]+v[None,:]-C)


"""
MAIN SISTA LOOP - with proximal gradient descent 
"""
def sista(p, q, hat_pi, D, beta0=None, u0=None, v0=None,
          rho=1e-1, gamma=1e-2, n_iters=500,
          sinkhorn_inner=1, tol=1e-7, verbose=False):

    N = p.shape[0]
    K = D.shape[0]

    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    hat_pi = hat_pi / hat_pi.sum()

    beta = np.zeros(K) if beta0 is None else beta0.astype(float).copy()
    u = np.zeros(N) if u0 is None else u0.astype(float).copy()
    v = np.zeros(N) if v0 is None else v0.astype(float).copy()

    prev_beta = beta.copy()

    for t in range(n_iters): #could do a while but assume don't know converges so need finite iterations
        # (1) cost from beta
        C = build_cost(beta, D)

        # (2) Sinkhorn updates: enforce marginals for current cost
        for _ in range(sinkhorn_inner):
            u, v = sinkhorn_one_pass(p, q, C, u, v)

        # (3) current plan from u,v,beta
        pi = compute_pi(u, v, C)

        # (4) gradient wrt beta_k: g_k = <hat_pi - pi, D[k]>
        diff = (hat_pi - pi)
        g = np.tensordot(D, diff, axes=([1, 2], [0, 1]))  # (K,)

        # (5) ISTA prox update
        z = beta - rho * g
        beta = soft_threshold(z, rho * gamma)

        # stopping criterion
        rel = np.linalg.norm(beta - prev_beta) / (np.linalg.norm(prev_beta) + 1e-12)
        if rel < tol:
            break
        prev_beta = beta

    return beta, u, v


"""
Sista with newton method 
Only step (5) changes: instead of a fixed step size rho for all coordinates,
each coordinate gets its own step size 1/H_kk from the Hessian diagonal.
"""
def sista_newton(p, q, hat_pi, D, beta0=None, u0=None, v0=None,
          gamma=1e-2, n_iters=500,
          sinkhorn_inner=1, tol=1e-7, mu=1e-6, verbose=False):

    N = p.shape[0]
    K = D.shape[0]

    p = p / p.sum()
    q = q / q.sum()
    hat_pi = hat_pi / hat_pi.sum()

    beta = np.zeros(K) if beta0 is None else beta0.astype(float).copy()
    u = np.zeros(N) if u0 is None else u0.astype(float).copy()
    v = np.zeros(N) if v0 is None else v0.astype(float).copy()

    prev_beta = beta.copy()

    for t in range(n_iters):
        # (1) cost from beta
        C = build_cost(beta, D)

        # (2) Sinkhorn updates
        for _ in range(sinkhorn_inner):
            u, v = sinkhorn_one_pass(p, q, C, u, v)

        # (3) current plan
        pi = compute_pi(u, v, C)

        # (4) gradient: g_k = <hat_pi - pi, D[k]>
        diff = (hat_pi - pi)
        g = np.tensordot(D, diff, axes=([1, 2], [0, 1]))  # (K,)

        # (5a) Hessian diagonal: H_kk = Var_pi(D[k]) = <pi, D[k]^2> - <pi, D[k]>^2
        D_flat = D.reshape(K, -1)         # (K, N*N)
        pi_flat = pi.ravel()              # (N*N,)
        mean_k = D_flat @ pi_flat         # (K,)  E_pi[D_k]
        mean_sq_k = (D_flat**2) @ pi_flat # (K,)  E_pi[D_k^2]
        H_diag = mean_sq_k - mean_k**2 + mu  # (K,) add mu for numerical stability

        # (5b) Proximal Newton update (coordinate-wise)
        z = beta - g / H_diag
        beta = soft_threshold(z, gamma / H_diag)

        # stopping criterion
        rel = np.linalg.norm(beta - prev_beta) / (np.linalg.norm(prev_beta) + 1e-12)
        if rel < tol:
            break
        prev_beta = beta

    return beta, u, v
