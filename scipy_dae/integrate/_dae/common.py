import numpy as np
from scipy.linalg import qr, solve_triangular


# TODO: Compare this with
# - ddassl.f by Petzold
# - epsode.f by Bryne and Hindmarsh
def select_initial_step(t0, y0, yp0, t_bound, rtol, atol, max_step):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    yp0 : ndarray, shape (n,)
        Initial value of the dependent variable's derivative.
    t_bound : float
        Final value of the independent variable.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.
    atol : float
        Desired absolute tolerance.
    max_step : float
        Maximum allowed step size.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] TODO: Find a reference.
    """
    min_step = 0.0
    threshold = atol / rtol
    hspan = abs(t_bound - t0)

    # compute an initial step size h using yp = y'(t0)
    wt = np.maximum(np.abs(y0), threshold)
    rh = 1.25 * np.linalg.norm(yp0 / wt, np.inf) / np.sqrt(rtol)
    h_abs = min(max_step, hspan)
    if h_abs * rh > 1:
        h_abs = 1 / rh
    h_abs = max(h_abs, min_step)
    return h_abs


# TODO:
# - make jac argument optional and use finite differences otherwise
# - Document this function and reference Shampine2002 accordingly
# - write tests for different problems (including fixed_y0 and fixed_yp0)
# - use sparse LU-decomposition if jac returns sparse matrices
# - use sparse QR-decomposition if available by scipy
def consistent_initial_conditions(fun, jac, t0, y0, yp0, fixed_y0=None, 
                                  fixed_yp0=None, rtol=1e-3, atol=1e-6, 
                                  *args):
    n = len(y0)
    
    if fixed_y0 is None:
        free_y = np.arange(n)
    else:
        # free_y = np.where(fixed_y0 == 0)[0]
        free_y = np.setdiff1d(np.arange(n), fixed_y0)
    
    if fixed_yp0 is None:
        free_yp = np.arange(n)
    else:
        # free_yp = np.where(fixed_yp0 == 0)[0]
        free_yp = np.setdiff1d(np.arange(n), fixed_yp0)
    
    if len(free_y) + len(free_yp) < n:
        raise ValueError(f"Too many components fixed, cannot solve the problem.")

    if not (isinstance(rtol, float) and rtol > 0):
        raise ValueError("Relative tolerance must be a positive scalar.")
    
    if rtol < 100 * np.finfo(float).eps:
        rtol = 100 * np.finfo(float).eps
        print(f"Relative tolerance increased to {rtol}")
    
    if np.any(np.array(atol) <= 0):
        raise ValueError("Absolute tolerance must be positive.")
    
    y0 = np.asarray(y0, dtype=float).reshape(-1)
    yp0 = np.asarray(yp0, dtype=float).reshape(-1)
    f = fun(t0, y0, yp0, *args)
    Jy, Jyp = jac(t0, y0, yp0)
    
    normf0 = np.linalg.norm(f)
    for _ in range(10):
        for _ in range(3):
            dy, dyp = solve_underdetermined_system(f, Jy, Jyp, free_y, free_yp)
            
            nrmv = max(np.linalg.norm(np.concatenate([y0, yp0])), np.linalg.norm(atol))
            nrmdv = np.linalg.norm(np.concatenate([dy, dyp]))
            
            if nrmdv > 2 * nrmv:
                factor = 2 * nrmv / nrmdv
                dy *= factor
                dyp *= factor
                nrmdv *= factor
            
            y0 += dy
            yp0 += dyp
            f = fun(t0, y0, yp0, *args)
            fnorm = np.linalg.norm(f)
            
            if (fnorm <= normf0) and (nrmdv <= 1e-3 * rtol * nrmv):
                return y0, yp0, f
        
        Jy, Jyp = jac(t0, y0, yp0)
    
    raise RuntimeError("Convergence failed.")


def solve_underdetermined_system(f, Jy, Jyp, free_y, free_yp):
    """Solve the underdetermined system 
        0 = f + Jy @ Delta_y + Jyp @ Delta_yp
    A solution is obtained with as many components as possible of 
    (transformed) Delta_y and Delta_yp set to zero.
    """
    n = len(f)
    Delta_y = np.zeros(n)
    Delta_yp = np.zeros(n)

    fixed = (n - len(free_y)) + (n - len(free_yp))
    if len(free_y) == 0:
        # solve 0 = f + Jyp @ Delta_yp
        rank, Q, R, p = qrank(Jyp)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        # Delta_yp[free_yp] = np.linalg.solve(R, d)
        Delta_yp[free_yp] = solve_triangular(R, d)[p]
        return Delta_y, Delta_yp
    
    if len(free_yp) == 0:
        # solve 0 = f + Jy @ Delta_y
        rank, Q, R, p = qrank(Jy)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        # Delta_y[free_y] = np.linalg.solve(R, d)
        Delta_y[free_y] = solve_triangular(R, d)[p]
        return Delta_y, Delta_yp
    
    # eliminate variables that are not free
    Jy = Jy[:, free_y]
    Jyp = Jyp[:, free_yp]
    
    # QR-decomposition of Fyp leads to the system
    # [R11, R12] [w1'] + [S11, S12] [w1] = [d1]
    # [  0,   0] [w2'] + [S21, S22] [w2] = [d2]
    # with S = Q.T @ Fy
    rank, Q, R, p = qrank(Jyp)
    d = -Q.T @ f
    if rank == n:
        # Full rank (ODE) case: 
        # Set all free dy to zero and solve triangular system below
        Delta_y[free_y] = 0
        Delta_yp[free_yp] = solve_triangular(R, d)[p]
    else:
        # Rank deficient (DAE) case:
        S = Q.T @ Jy
        rankS, QS, RS, pS = qrank(S[rank:])
        rankdef = n - (rank + rankS)
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")
        
        # decompose d
        d1 = d[:rank]
        d2 = d[rank:]

        # compute basic solution of underdetermined system
        # [S21, S22] [w1] = d2
        #            [w2]
        # using column pivoting QR-decomposition
        w = np.zeros(RS.shape[1])
        w[:rankS] = solve_triangular(RS[:rankS, :rankS], (QS.T @ d2[:rankS]))
        w = w[pS]

        # set w2' = 0 and solve the remaining system
        # [R11] w1' = d1 - [S11, S12] [w1]
        #                             [w2]
        # w1p = R(1:rank,1:rank) \ (d(1:rank) - S(1:rank,:) * w);
        # wp1 = solve_triangular(R[:rank, :rank], d[:rank] - S[:rank] @ w)
        # wp_ = np.concatenate([wp1, np.zeros(len(free_yp) - rank)])
        # wp = np.zeros(n)
        # wp = wp_[p]
        wp = np.zeros(R.shape[1])
        if rank > 0:
            wp[:rank] = solve_triangular(R[:rank, :rank], d1 - S[:rank] @ w)
        wp = wp[p]

        # store w and wp
        Delta_y[free_y] = w
        Delta_yp[free_yp] = wp
    
    return Delta_y, Delta_yp


def qrank(A):
    Q, R, p = qr(A, pivoting=True)
    tol = max(A.shape) * np.finfo(float).eps * abs(R[0, 0])
    rank = np.sum(abs(np.diag(R)) > tol)
    return rank, Q, R, p
