from itertools import groupby
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
        free_y = np.where(fixed_y0 == 0)[0]
    
    if fixed_yp0 is None:
        free_yp = np.arange(n)
    else:
        free_yp = np.where(fixed_yp0 == 0)[0]
    
    if len(free_y) + len(free_yp) < n:
        raise ValueError(f"Too many components fixed, cannot solve the problem.")

    if not (isinstance(rtol, float) and rtol > 0):
        raise ValueError("Relative tolerance must be a positive scalar.")
    
    if rtol < 100 * np.finfo(float).eps:
        rtol = 100 * np.finfo(float).eps
        print(f"Relative tolerance increased to {rtol}")
    
    if np.any(np.array(atol) <= 0):
        raise ValueError("Absolute tolerance must be positive.")
    
    y0 = np.asarray(y0).flatten()
    yp0 = np.asarray(yp0).flatten()
    f = fun(t0, y0, yp0, *args)
    Jy, Jyp = jac(t0, y0, yp0, f)
    
    normf0 = np.linalg.norm(f)
    for _ in range(10):
        for _ in range(3):
            dy, dyp = solve_underdetermined_system(f, Jy, Jyp, n, free_y, free_yp)
            
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
                return y0, yp0, fnorm
        
        Jy, Jyp = jac(t0, y0, yp0, f)
    
    raise RuntimeError("Convergence failed.")


class DaeSolution:
    """Continuous DAE solution.

    It is organized as a collection of `DenseOutput` objects which represent
    local interpolants. It provides an algorithm to select a right interpolant
    for each given point.

    The interpolants cover the range between `t_min` and `t_max` (see
    Attributes below). Evaluation outside this interval is not forbidden, but
    the accuracy is not guaranteed.

    When evaluating at a breakpoint (one of the values in `ts`) a segment with
    the lower index is selected.

    Parameters
    ----------
    ts : array_like, shape (n_segments + 1,)
        Time instants between which local interpolants are defined. Must
        be strictly increasing or decreasing (zero segment with two points is
        also allowed).
    interpolants : list of DenseOutput with n_segments elements
        Local interpolants. An i-th interpolant is assumed to be defined
        between ``ts[i]`` and ``ts[i + 1]``.
    alt_segment : boolean
        Requests the alternative interpolant segment selection scheme. At each
        solver integration point, two interpolant segments are available. The
        default (False) and alternative (True) behaviours select the segment
        for which the requested time corresponded to ``t`` and ``t_old``,
        respectively. This functionality is only relevant for testing the
        interpolants' accuracy: different integrators use different
        construction strategies.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """
    def __init__(self, ts, interpolants, alt_segment=False):
        ts = np.asarray(ts)
        d = np.diff(ts)
        # The first case covers integration on zero segment.
        if not ((ts.size == 2 and ts[0] == ts[-1])
                or np.all(d > 0) or np.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.side = "right" if alt_segment else "left"
            self.ts_sorted = ts
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.side = "left" if alt_segment else "right"
            self.ts_sorted = ts[::-1]

    def _call_single(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # if alt_segment=False, then we prioritize a segment with a lower
        # index.
        ind = np.searchsorted(self.ts_sorted, t, side=self.side)

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - 1 - segment

        return self.interpolants[segment](t)

    def __call__(self, t):
        """Evaluate the solution.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate at.

        Returns
        -------
        y : ndarray, shape (n_states,) or (n_states, n_points)
            Computed values. Shape depends on whether `t` is a scalar or a
            1-D array.
        """
        t = np.asarray(t)

        if t.ndim == 0:
            return self._call_single(t)

        order = np.argsort(t)
        reverse = np.empty_like(order)
        reverse[order] = np.arange(order.shape[0])
        t_sorted = t[order]

        # See comment in self._call_single.
        segments = np.searchsorted(self.ts_sorted, t_sorted, side=self.side)
        segments -= 1
        segments[segments < 0] = 0
        segments[segments > self.n_segments - 1] = self.n_segments - 1
        if not self.ascending:
            segments = self.n_segments - 1 - segments

        ys = []
        yps = []
        group_start = 0
        for segment, group in groupby(segments):
            group_end = group_start + len(list(group))
            y, yp = self.interpolants[segment](t_sorted[group_start:group_end])
            ys.append(y)
            yps.append(yp)
            group_start = group_end

        ys = np.hstack(ys)
        ys = ys[:, reverse]
        yps = np.hstack(yps)
        yps = yps[:, reverse]

        return ys, yps


def solve_underdetermined_system(f, Jy, Jyp, n, free_y, free_yp):
    """Solve the underdetermined system 
        0 = f + Jyp @ Delta_yp + Jy @ Delta_y
    A solution is obtained with as many components as possible of 
    (transformed) Delta_yp and Delta_y set to zero.
    """
    Delta_y = np.zeros(n)
    Delta_yp = np.zeros(n)
    
    fixed = (n - len(free_y)) + (n - len(free_yp))
    
    if len(free_y) == 0:
        rank, Q, R, p = qrank(Jyp)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        Delta_yp[free_yp] = np.linalg.solve(R, d)
        return Delta_y, Delta_yp
    
    if len(free_yp) == 0:
        rank, Q, R, p = qrank(Jy)
        rankdef = n - rank
        if rankdef > 0:
            if rankdef <= fixed:
                raise ValueError(f"Too many fixed components, rank deficiency is {rankdef}.")
            else:
                raise ValueError("Index greater than one.")
        d = -Q.T @ f
        Delta_y[free_y] = np.linalg.solve(R, d)
        return Delta_y, Delta_yp
    
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
        Delta_yp[free_yp] = np.linalg.solve(R, d)[p]
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

        # compute basic solution of underdetermined system
        # [S21, S22] [w1] = d2
        #            [w2]
        # using column pivoting QR-decomposition
        d2 = d[rank:]
        w_ = np.zeros(n)
        w_[:rankS] = solve_triangular(RS[:rankS, :rankS], (QS.T @ d2[:rankS]))
        w = np.zeros(n)
        w[pS] = w_

        # set w2' = 0 and solve the remaining system
        # [R11] w1' = d1 - [S11, S12] [w1]
        #                             [w2]
        w1p = solve_triangular(R[:rank, :rank], d[:rank] - S[:rank] @ w)
        wp_ = np.concatenate([w1p, np.zeros(len(free_yp) - rank)])
        wp = np.zeros(n)
        wp[p] = wp_

        # store w and wp
        Delta_y[free_y] = w
        Delta_yp[free_yp] = wp
    
    return Delta_y, Delta_yp


def qrank(A):
    Q, R, E = qr(A, pivoting=True)
    tol = max(A.shape) * np.finfo(float).eps * abs(R[0, 0])
    rank = np.sum(abs(np.diag(R)) > tol)
    return rank, Q, R, E
