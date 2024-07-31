import numpy as np
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

# Butcher Tableau coefficients of the method
S2 = np.sqrt(2)
gamma = 2 - S2
d = gamma / 2
w = S2 / 4

# Butcher Tableau coefficients for the error estimator
e0 = (1 - w) / 3
e1 = w + (1 / 3)
e2 = d / 3

# embedded implicit method
b1_hat = (1 - w) / 3
# b1_hat = d
b2_hat = w + 1 / 3
b3_hat = d / 3
# b4_hat = d

# b23_hat = np.linalg.solve(
#     np.array([
#         [   gamma, 1],
#         [gamma**2, 1],
#     ]),
#     np.array([
#         1 / 2 - b1_hat - b4_hat,
#         1 / 3 - b4_hat,
#     ])
# )

# b2_hat, b3_hat = b23_hat

# compute embedded method for error estimate
c_hat = np.array([0, gamma, 1])
vander = np.vander(c_hat, increasing=True).T

rhs = 1 / np.arange(1, len(c_hat))
gamma0 = 1 / d
# gamma0 = d
b0 = gamma0
rhs[0] -= b0
rhs -= gamma0

b_hat = np.linalg.solve(vander[:-1, 1:], rhs)
# b_hat = np.array([b0, *b_hat])
# b = np.array([w, w, d])
b = np.array([w, d])
v = b - b_hat

# Coefficients required for interpolating y'
pd0 = 1.5 + S2
pd1 = 2.5 + 2 * S2
pd2 = -(6 + 4.5 * S2)

# TODO: 
# - documentation
def solve_trbdf2_system(fun, z0, LU, solve_lu, scale, tol):
    z = z0.copy()
    dz_norm_old = None
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        f = fun(z)
        if not np.all(np.isfinite(f)):
            break

        dz = solve_lu(LU, -f)
        dz_norm = norm(dz / scale)
        if dz_norm_old is not None:
            rate = dz_norm / dz_norm_old

        if (rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dz_norm > tol)):
            break

        z += dz

        if (dz_norm == 0 or rate is not None and rate / (1 - rate) * dz_norm < tol):
            converged = True
            break

        dz_norm_old = dz_norm

    return converged, k + 1, z, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    """Predict by which factor to increase/decrease the step size.

    The algorithm is described in [1]_.

    Parameters
    ----------
    h_abs, h_abs_old : float
        Current and previous values of the step size, `h_abs_old` can be None
        (see Notes).
    error_norm, error_norm_old : float
        Current and previous values of the error norm, `error_norm_old` can
        be None (see Notes).
    s : int
        Number of stages of the Radau IIA method.

    Returns
    -------
    factor : float
        Predicted factor.

    Notes
    -----
    If `h_abs_old` and `error_norm_old` are both not None then a two-step
    algorithm is used, otherwise a one-step algorithm is used.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    """
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / 3)

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** (-1 / 3)

    return factor


# TODO:
# - adapt documentation
class TRBDF2DAE(DaeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 2s - 1.

    The implementation follows [4]_, where most of the ideas come from [2]_. 
    The embedded formula of [3]_ is applied to implicit differential equations.
    The error is controlled with a (s)th-order accurate embedded formula as 
    introduced in [2]_ and refined in [3]_. The procedure is slightly adapted 
    by [4]_ to cope with implicit differential equations. The embedded error 
    estimate can be mixed with the contunous error of the lower order 
    collocation polynomial as porposed in [5]_ and [6]_.
    
    A cubic polynomial 
    which satisfies the collocation conditions is used for the dense output of 
    both state and derivatives.

    Parameters
    ----------
    fun : callable
        Function defining the DAE system: ``f(t, y, yp) = 0``. The calling 
        signature is ``fun(t, y, yp)``, where ``t`` is a scalar and 
        ``y, yp`` are ndarrays with 
        ``len(y) = len(yp) = len(y0) = len(yp0)``. ``fun`` must return 
        an array of the same shape as ``y, yp``. See `vectorized` for more
        information.
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    yp0 : array_like, shape (n,)
        Initial derivative.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    stages : int, optional
        Number of used stages. Default is 3, which corresponds to the 
        ``solve_ivp`` method. Only odd number of stages are allowed.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. HHere `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    continuous_error_weight : float, optional
        Weighting of continuous error of the dense output as introduced in 
        [5]_ and [6]_. The embedded error is weighted by (1 - continuous_error_weight). 
        Has to satisfy 0 <= continuous_error_weight <= 1. Default is 0.0, i.e., only 
        the embedded error is considered.
    jac : {None, array_like, sparse_matrix, callable}, optional
        Jacobian matrix of the right-hand side of the system with respect to
        y, required by this method. The Jacobian matrix has shape (n, n) and
        its element (i, j) is equal to ``d f_i / d y_j``.
        There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)`` as necessary.
              For the 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    # TODO: Adapt and test this.
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` 
        and ``yp`` of shape ``(n,)``, where ``n = len(y0) = len(yp0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` and ``yp`` of 
        shape ``(n, k)``, where ``k`` is an integer. In this case, `fun` must 
        behave such that ``fun(t, y, yp)[:, i] == fun(t, y[:, i], yp[:, i])``.

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by this method, but may result in slower
        execution overall in some circumstances (e.g. small ``len(y0)``).
        Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    yp : ndarray
        Current derivative.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.

    References
    ----------
    .. [1] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [2] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    .. [3] J. de Swart, G. Söderlind, "On the construction of error estimators for 
           implicit Runge-Kutta methods", Journal of Computational and Applied 
           Mathematics, 86, pp. 347-358, 1997.
    .. [4] B. Fabien, "Analytical System Dynamics: Modeling and Simulation", 
           Sec. 5.3.5.
    .. [5] N. Guglielmi, E. Hairer, "Implementing Radau IIA Methods for Stiff 
           Delay Differential Equations", Computing 67, 1-12, 2001.
    .. [6] N. Guglielmi, "Open issues in devising software for the numerical 
           solution of implicit delay differential equations", Journal of 
           Computational and Applied Mathematics 185, 261-277, 2006.
    """
    def __init__(self, fun, t0, y0, yp0, t_bound, max_step=np.inf, 
                 rtol=1e-3, atol=1e-6, jac=None, 
                 jac_sparsity=None, vectorized=False, 
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, first_step, max_step, vectorized, jac, jac_sparsity)
        self.y_old = None

        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None

        self.current_jac = True
        self.LU = None

        # variables for implementing first-same-as-last
        self.z_scaled = self.yp

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp
        f = self.f

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            error_norm_old = self.error_norm_old

        Jy = self.Jy
        Jyp = self.Jyp
        LU = self.LU
        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h
            # print(f"t_new: {t_new}")

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            # we iterate on the variable z = h yp, as explained in [1]
            z0 = h_abs * self.z_scaled
            scale = atol + np.abs(y) * rtol

            # TODO: Better initial guess for z0 as explained by Hosea?

            # TR stage
            converged_tr = False
            while not converged_tr:
                if LU is None:
                    LU = self.lu(Jy + Jyp / (d * h))

                t_gamma = t + h * gamma
                fun_tr = lambda z: self.fun(t_gamma, y + z, z / (d * h) - yp)
                converged_tr, n_iter_tr, z_tr, rate_tr = solve_trbdf2_system(fun_tr, z0, LU, self.solve_lu, scale, self.newton_tol)

                if not converged_tr:
                    if current_jac:
                        break
                    Jy, Jyp = self.jac(t, y, yp, f)
                    LU = None
                    current_jac = True

            if not converged_tr:
                h_abs *= 0.5
                LU = None
                continue

            y_tr = y + z_tr
            yp_tr = z_tr / (h * d) - yp

            # BDF stage
            z_bdf0 = pd0 * z0 + pd1 * z_tr + pd2 * (y_tr - y)
            converged_bdf = False
            while not converged_bdf:
                if LU is None:
                    LU = self.lu(Jy + Jyp / (d * h))

                fun_bdf = lambda z: self.fun(t_new, y + z, z / (h * d) - (w / d) * (yp + yp_tr))
                converged_bdf, n_iter_bdf, z_bdf, rate_bdf = solve_trbdf2_system(fun_bdf, z_bdf0, LU, self.solve_lu, scale, self.newton_tol)

                if not converged_bdf:
                    if current_jac:
                        break
                    Jy, Jyp = self.jac(t, y, yp, f)
                    LU = None
                    current_jac = True

            if not converged_bdf:
                h_abs *= 0.5
                LU = None
                continue

            n_iter = max(n_iter_tr, n_iter_bdf)
            rate = max(rate_tr, rate_bdf)
            # if rate_tr is not None:
            #     if rate_bdf is not None:
            #         rate = max(rate_tr, rate_bdf)
            #     else:
            #         rate = rate_tr
            # else:
            #     rate = 0

            y_new = y + z_bdf
            yp_new = z_bdf / (h * d) - (w / d) * (yp + yp_tr)

            # error = 0.5 * (y + e0 * z0 + e1 * z_tr + e2 * z_bdf - y_new)
            error = h * ((b1_hat - w) * yp + (b2_hat - w) * yp_tr + (b3_hat - d) * yp_new)
            # error = self.solve_lu(LU, error) #* (d * h) #* d / 3

            # # implicit error estimate
            # # yp_hat_new = (y_new - y) / (b4_hat * h) - (b1_hat / b4_hat) * yp - (b2_hat / b4_hat) * yp_tr - (b3_hat / b4_hat) * yp_new
            # yp_hat_new = (v @ np.array([yp_tr, yp_new]) - b0 * yp) * d
            # F = self.fun(t_new, y_new, yp_hat_new)
            # error = self.solve_lu(LU, -F)
            # error = h * d * (v @ np.array([yp_tr, yp_new]) - b0 * yp - yp_new / d)

            # # # error_Fabien = h * MU_REAL * (v @ Yp - b0 * yp - yp_new / MU_REAL)
            # # yp_hat_new = MU_REAL * (v @ Yp - b0 * yp)
            # # F = self.fun(t_new, y_new, yp_hat_new)
            # # error_Fabien = self.solve_lu(LU_real, -F)

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            
            # TODO: Add stabilized error
            # stabilised_error = self.solve_lu(LU, error)
            stabilised_error = error
            error_norm = norm(stabilised_error / scale)

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU = None
            else:
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU = None

        f_new = self.fun(t_new, y_new, yp_new)
        if recompute_jac:
            Jy, Jyp = self.jac(t_new, y_new, yp_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y
        self.yp_old = yp

        self.t = t_new
        self.y = y_new
        self.yp = yp_new
        self.f = f_new

        self.LU = LU
        self.current_jac = current_jac
        self.Jy = Jy
        self.Jyp = Jyp
        self.z_scaled = z_bdf / self.h_abs_old

        self.t_old = t

        # variables for dense output
        self.h_old = self.h_abs_old * self.direction
        self.z0 = z0
        self.z_tr = z_tr
        self.z_bdf = z_bdf
        self.y_tr = y_tr

        return step_accepted, message

    def _dense_output_impl(self):
        return Trbdf2DenseOutput(self.t_old, self.t, self.h_old,
                                 self.z0, self.z_tr, self.z_bdf,
                                 self.y_old, self.y_tr, self.y)


class Trbdf2DenseOutput(DenseOutput):
    """
    The dense output formula as described in _[1] depends on the time at which interpolation is done.
    Specifically, the formula has two sets of coefficients which are applicable depending on
    whether t <= t0 + gamma*h or t > t0 + gamma*h

    .. [1] Hosea, M. E., and L. F. Shampine. "Analysis and implementation of TR-BDF2."
           Applied Numerical Mathematics 20.1-2 (1996): 21-37.
    """

    def __init__(self, t0, t, h, z0, z_tr, z_bdf, y0, y_tr, y_bdf):
        super().__init__(t0, t)
        self.t0 = t0
        self.h = h
        # coefficients for the case t <= t_old + gamma*h (denoted as v10, v11...)
        v10 = y0
        v11 = gamma * z0
        v12 = y_tr - y0 - v11
        v13 = gamma * (z_tr - z0)
        self.V1 = np.array([v10, v11, 3 * v12 - v13, v13 - 2 * v12])

        # coefficients for the case t > t0 + gamma*h
        v20 = y_tr
        v21 = (1 - gamma) * z_tr
        v22 = y_bdf - y_tr - v21
        v23 = (1 - gamma) * (z_bdf - z_tr)
        self.V2 = np.array([v20, v21, 3 * v22 - v23, v23 - 2 * v22])

    def _call_impl(self, t):
        if t.ndim == 0:
            if t <= self.t0 + gamma * self.h:
                r = (t - self.t0) / (gamma * self.h)
                V = self.V1
            else:
                t_tr = self.t0 + gamma * self.h
                V = self.V2
                r = (t - t_tr) / ((1 - gamma) * self.h)

            R = np.cumprod([1, r, r, r])
            y = np.dot(V.T, R)
        else:
            y = []
            for tk in t:
                y.append(self._call_impl(tk))
            y = np.array(y).T
        return y
