import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.linalg import eig, cdf2rdf
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


def butcher_tableau(s):
    # nodes are given by the zeros of the Radau polynomial, see Hairer1999 (7)
    poly = Poly([0, 1]) ** (s - 1) * Poly([-1, 1]) ** s
    poly_der = poly.deriv(s - 1)
    c = poly_der.roots()

    # compute coefficients a_ij, see Hairer1999 (11)
    A = np.zeros((s, s))
    for i in range(s):
        Mi = np.zeros((s, s))
        ri = np.zeros(s)
        for q in range(s):
            Mi[q] = c**q
            ri[q] = c[i] ** (q + 1) / (q + 1)
        A[i] = np.linalg.solve(Mi, ri)

    b = A[-1, :]
    p = 2 * s - 1
    return A, b, c, p


def radau_constants(s):
    # Butcher tableau
    A, b, c, p = butcher_tableau(s)

    # inverse coefficient matrix
    A_inv = np.linalg.inv(A)

    # eigenvalues and corresponding eigenvectors of inverse coefficient matrix
    lambdas, V = eig(A_inv)

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    V = V[:, idx]

    # scale eigenvectors to get a "nice" transformation matrix (used by original 
    # scipy code) and at the same time minimizes the condition number of V
    for i in range(s):
        V[:, i] /= V[-1, i]

    # convert complex eigenvalues and eigenvectors to real eigenvalues 
    # in a block diagonal form and the associated real eigenvectors
    Mus, T = cdf2rdf(lambdas, V)
    TI = np.linalg.inv(T)

    # check if everything worked
    assert np.allclose(TI @ A_inv @ T, Mus)

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag
    
    # compute embedded method for error estimate
    c_hat = np.array([0, *c])
    vander = np.vander(c_hat, increasing=True).T

    rhs = 1 / np.arange(1, s + 1)
    gamma0 = 1 / gammas[0] # real eigenvalue of A, i.e., 1 / gamma[0]
    b0 = gamma0 # note: this leads to the formula implemented inr ride.m by Fabien
    # b0 = 0.02 # proposed value of de Swart
    rhs[0] -= b0
    rhs -= gamma0

    b_hat = np.linalg.solve(vander[:-1, 1:], rhs)
    v = b - b_hat

    # Compute the inverse of the Vandermonde matrix to get the interpolation matrix P.
    P = np.linalg.inv(vander)[1:, 1:]

    # Compute coefficients using Vandermonde matrix.
    vander2 = np.vander(c, increasing=True)
    P2 = np.linalg.inv(vander2)

    # These linear combinations are used in the algorithm.
    MU_REAL = gammas[0]
    MU_COMPLEX = alphas -1j * betas
    TI_REAL = TI[0]
    TI_COMPLEX = TI[1::2] + 1j * TI[2::2]

    return A_inv, c, T, TI, P, P2, b0, v, MU_REAL, MU_COMPLEX, TI_REAL, TI_COMPLEX, b_hat


def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu,
                             C, T, TI, A_inv):
    """Solve the collocation system.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    h : float
        Step to try.
    Z0 : ndarray, shape (s, n)
        Initial guess for the solution. It determines new values of `y` at
        ``t + h * C`` as ``y + Z0``, where ``C`` is the Radau method constants.
    scale : ndarray, shape (n)
        Problem tolerance scale, i.e. ``rtol * abs(y) + atol``.
    tol : float
        Tolerance to which solve the system. This value is compared with
        the normalized by `scale` error.
    LU_real, LU_complex
        LU decompositions of the system Jacobians.
    solve_lu : callable
        Callable which solves a linear system given a LU decomposition. The
        signature is ``solve_lu(LU, b)``.
    C : ndarray, shape (s,)
        Array containing the Radau IIA nodes.
    T, TI : ndarray, shape (s, s)
        Transformation matrix and inverse of the methods coefficient matrix A.
    A_inv : ndarray, shape (s, s)
        Inverse the methods coefficient matrix A.

    Returns
    -------
    converged : bool
        Whether iterations converged.
    n_iter : int
        Number of completed iterations.
    Z : ndarray, shape (3, n)
        Found solution.
    rate : float
        The rate of convergence.
    """
    s, n = Z0.shape
    ncs = s // 2
    tau = t + h * C

    Z = Z0
    W = TI.dot(Z0)
    Yp = (A_inv / h) @ Z
    Y = y + Z

    F = np.empty((s, n))

    dW_norm_old = None
    dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(s):
            F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        U = TI @ F
        f_real = -U[0]
        f_complex = np.empty((ncs, n), dtype=complex)
        for i in range(ncs):
            f_complex[i] = -(U[2 * i + 1] + 1j * U[2 * i + 2])

        dW_real = solve_lu(LU_real, f_real)
        dW_complex = np.empty_like(f_complex)
        for i in range(ncs):
            dW_complex[i] = solve_lu(LU_complex[i], f_complex[i])

        dW[0] = dW_real
        for i in range(ncs):
            dW[2 * i + 1] = dW_complex[i].real
            dW[2 * i + 2] = dW_complex[i].imag

        dW_norm = norm(dW / scale)
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        if (rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
            break

        W += dW
        Z = T.dot(W)
        Yp = (A_inv / h) @ Z
        Y = y + Z

        if (dW_norm == 0 or rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        dW_norm_old = dW_norm

    return converged, k + 1, Y, Yp, Z, rate


def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s):
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
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / (s + 1))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

    return factor


class RadauDAE(DaeSolver):
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
    def __init__(self, fun, t0, y0, yp0, t_bound, stages=3,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, 
                 continuous_error_weight=0.0, jac=None, 
                 jac_sparsity=None, vectorized=False, 
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, first_step, max_step, vectorized, jac, jac_sparsity)
        self.y_old = None

        # # TODO: Manipulate rtol and atol according to Radau.f?
        # # TODO: What is the idea behind this?
        # # https://github.com/luchr/ODEInterface.jl/blob/0bd134a5a358c4bc13e0fb6a90e27e4ee79e0115/src/radau5.f#L399-L421
        # print(f"rtol: {rtol}")
        # print(f"atol: {atol}")
        # expm = 2 / 3
        # quot = atol / rtol
        # rtol = 0.1 * rtol**expm
        # atol = rtol * quot
        # print(f"rtol: {rtol}")
        # print(f"atol: {atol}")

        assert stages % 2 == 1
        self.stages = stages
        self.A_inv, self.C, self.T, self.TI, self.P, self.P2, self.b0, self.v, self.MU_REAL, self.MU_COMPLEX, self.TI_REAL, self.TI_COMPLEX, self.b_hat = radau_constants(stages)

        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        assert 0 <= continuous_error_weight <= 1
        self.continuous_error_weight = continuous_error_weight
        self.sol = None

        self.current_jac = True
        self.LU_real = None
        self.LU_complex = None
        self.Z = None

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp
        f = self.f

        s = self.stages
        MU_REAL = self.MU_REAL
        MU_COMPLEX = self.MU_COMPLEX
        C = self.C
        T = self.T
        TI = self.TI
        A_inv = self.A_inv
        v = self.v
        b0 = self.b0
        P2 = self.P2

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
        LU_real = self.LU_real
        LU_complex = self.LU_complex

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

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            if self.sol is None:
                Z0 = np.zeros((s, y.shape[0]))
            else:
                Z0 = self.sol(t + h * C).T - y
            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    # Fabien (5.59) and (5.60)
                    LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                    LU_complex = [self.lu(MU / h * Jyp + Jy) for MU in MU_COMPLEX]

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU_real, LU_complex, self.solve_lu,
                    C, T, TI, A_inv)

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            # Hairer1996 (8.2b)
            y_new = Y[-1]
            yp_new = Yp[-1]

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol

            ############################################
            # error of collocation polynomial of order s
            ############################################
            error_collocation = y - P2[0] @ Y
            
            ###############
            # Fabien (5.65)
            ###############
            # note: This is the embedded method that is stabilized below
            # TODO: Store MU_REAL * v during construction.
            # error_Fabien = h * MU_REAL * (v @ Yp - b0 * yp - yp_new / MU_REAL)
            yp_hat_new = MU_REAL * (v @ Yp - b0 * yp)
            F = self.fun(t_new, y_new, yp_hat_new)
            error_Fabien = self.solve_lu(LU_real, -F)

            # # add another Newton step for stabilization
            # # TODO: This is definitely better for the pendulum problem. I think for the error above
            # #       R(z) = -1 for hz -> intfy and the error below satisfies R(z) = 0 for hz -> infty
            # y_hat_new = y_new + error_Fabien
            # # yp_hat_new = MU_REAL / h * (y_hat_new - y - h * b0 * yp - h * b_hat @ Yp)
            # yp_hat_new = MU_REAL * (error_Fabien / h - b0 * yp + v @ Yp)
            # F = self.fun(t_new, y_hat_new, yp_hat_new)
            # error_Fabien += self.solve_lu(LU_real, -F)
            # # y_hat_new += self.solve_lu(LU_real, -F)
            # # error_Fabien = y_hat_new - y_new

            # mix embedded error with collocation error as proposed in Guglielmi2001/ Guglielmi2003
            error = (
                self.continuous_error_weight * np.abs(error_collocation)**((s + 1) / s) 
                + (1 - self.continuous_error_weight) * np.abs(error_Fabien)
            )
            error_norm = norm(error / scale)

            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            # if rejected and error_norm > 1: # try with stabilised error estimate
            # # if True:sol(t
            #     print(f"rejected")
            #     # add another Newton step for stabilization
            #     # TODO: This is definitely better for the pendulum problem. I think for the error above
            #     #       R(z) = -1 for hz -> intfy and the error below satisfies R(z) = 0 for hz -> infty
            #     y_hat_new = y_new + error_Fabien
            #     yp_hat_new = MU_REAL / h * (y_hat_new - y - h * b0 * yp - h * b_hat @ Yp)
            #     # yp_hat_new = MU_REAL * (error_Fabien / h - b0 * yp + v @ Yp)
            #     F = self.fun(t_new, y_hat_new, yp_hat_new)
            #     error_Fabien = self.solve_lu(LU_real, -F)

            #     # mix embedded error with collocation error as proposed in Guglielmi2001/ Guglielmi2003
            #     error = (
            #         self.continuous_error_weight * np.abs(error_collocation)**((s + 1) / s) 
            #         + (1 - self.continuous_error_weight) * np.abs(error_Fabien)
            #     )                
            #     error_norm = norm(error / scale)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        # Step is converged and accepted
        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old, s)
        factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU_real = None
            LU_complex = None

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

        self.Z = Z

        self.LU_real = LU_real
        self.LU_complex = LU_complex
        self.current_jac = current_jac
        self.Jy = Jy
        self.Jyp = Jyp

        self.t_old = t
        self.sol = self._compute_dense_output()

        return step_accepted, message

    def _compute_dense_output(self):
        Q = np.dot(self.Z.T, self.P)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)

    def _dense_output_impl(self):
        return self.sol


class RadauDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        x = np.atleast_1d(x)
        p = np.tile(x, (self.order + 1, 1))
        p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake.
        y = np.dot(self.Q, p)
        y += self.y_old[:, None]
        if t.ndim == 0:
            y = np.squeeze(y)

        return y
