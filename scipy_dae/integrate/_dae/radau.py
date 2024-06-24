import numpy as np
from numpy.polynomial import Polynomial as Poly
from scipy.linalg import eig, cdf2rdf
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver
from .base import DAEDenseOutput as DenseOutput


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
    print(f"A:\n{A}")

    # inverse coefficient matrix
    A_inv = np.linalg.inv(A)
    print(f"A_inv:\n{A_inv}")

    # eigenvalues and corresponding eigenvectors of inverse coefficient matrix
    lambdas, V = eig(A_inv)
    print(f"lambdas:\n{lambdas}")
    print(f"V:\n{V}")

    # sort eigenvalues and permut eigenvectors accordingly
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    V = V[:, idx]
    print(f"lambdas:\n{idx}")
    print(f"lambdas:\n{lambdas}")
    print(f"V:\n{V}")

    # scale eigenvectors to get a "nice" transformation matrix (used by scipy)
    # and at the same time minimizes the condition number of V
    for i in range(s):
        V[:, i] /= V[-1, i]

    # # scale only the first two eigenvectors (used by Hairer)
    # V[:, 0] /= V[-1, 0]
    # V[:, 1] /= V[-1, 1]

    print(f"V scaled:\n{V}")

    # convert complex eigenvalues and eigenvectors to real eigenvalues 
    # in a block diagonal form and the associated real eigenvectors
    lambdas_real, V_real = cdf2rdf(lambdas, V)
    print(f"lambdas_real:\n{lambdas_real}")
    print(f"V_real:\n{V_real}")

    # transform to get scipy's/ Hairer's ordering
    R = np.fliplr(np.eye(s))
    R = np.eye(s)
    Mus = R @ lambdas_real @ R.T
    print(f"R:\n{R}")
    print(f"Mus:\n{Mus}")

    T = V_real @ R.T
    TI = np.linalg.inv(T)
    print(f"T:\n{T}")

    # check if everything worked
    assert np.allclose(TI @ A_inv @ T, Mus)

    # extract real and complex eigenvalues
    real_idx = lambdas.imag == 0
    complex_idx = ~real_idx
    gammas = lambdas[real_idx].real
    alphas_betas = lambdas[complex_idx]
    alphas = alphas_betas[::2].real
    betas = alphas_betas[::2].imag
    print(f"gammas: {gammas}")
    print(f"alphas: {alphas}")
    print(f"betas: {betas}")
    
    # compute embedded method for error estimate,
    # see https://arxiv.org/abs/1306.2392 equation (10)
    # # TODO: This is not correct yet, see also here: https://math.stackexchange.com/questions/4441391/embedding-methods-into-implicit-runge-kutta-schemes
    # TODO: This is correct, document this extended tableau!
    c_hat = np.array([0, *c])
    print(f"c_hat: {c_hat}")

    vander = np.vander(c_hat, increasing=True).T
    print(f"vander:\n{vander}")

    rhs = 1 / np.arange(1, s + 1)
    print(f"rhs:\n{rhs}")

    gamma0 = 1 / gammas[0] # real eigenvalue of A, i.e., 1 / gamma[0]
    b0 = gamma0 # note: this leads to the formula of Fabien!
    # b0 = 0.02
    # b0 *= 0.07275668505489
    # b0 *= 0.9
    rhs[0] -= b0
    rhs -= gamma0
    print(f"rhs:\n{rhs}")

    b_hat = np.linalg.solve(vander[:-1, 1:], rhs)
    print(f"b_hat:\n{b_hat}")

    rhs = 1 / np.arange(1, s + 2)

    E = (b_hat - b) @ A_inv
    E /= gamma0
    print(f"c: {c}")
    print(f"E:\n{E}")

    # used by Fabien2009
    v = np.array([0.428298294115369, -0.245039074384917, 0.366518439460903])
    print(f"v:\n{v}")

    v = b - b_hat
    print(f"v2:\n{v}")

    # # collocation polynomial
    # V = np.vander(np.array([0, *C]), increasing=True).T
    # # print(f"V:\n{V}")

    # Compute the inverse of the Vandermonde matrix to get the interpolation matrix P
    P = np.linalg.inv(vander)[1:, 1:]
    print(f"P:\n{P}")

    # These linear combinations are used in the algorithm.
    MU_REAL = gammas[0]
    MU_COMPLEX = alphas -1j * betas
    TI_REAL = TI[0]
    TI_COMPLEX = TI[1::2] + 1j * TI[2::2]

    print(f"")
    return A_inv, c, T, TI, P, b0, v, MU_REAL, MU_COMPLEX, TI_REAL, TI_COMPLEX, Mus, b_hat
    # return A, A_inv, b, c, T, TI, E, P, b_hat, gamma0, b0, v, p, gammas, alphas, betas, MU_REAL, MU_COMPLEX, TI_REAL, TI_COMPLEX


# s = 1
s = 3
# s = 5
# s = 7
# s = 9
assert s % 2 == 1
ncs = s // 2 # number of conjugate complex eigenvalues
A_inv, C, T, TI, P, b0, v, MU_REAL, MU_COMPLEX, TI_REAL, TI_COMPLEX, Mus, b_hat = radau_constants(s)

# S6 = 6 ** 0.5

# # Butcher tableau. A is not used directly, see below.
# C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
# E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
# print(f"E:\n{E}")

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
# MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
# MU_REAL = gammas[0]
# MU_COMPLEX = alphas -1j * betas

# # These are transformation matrices.
# T = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0]])
# TI = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# These linear combinations are used in the algorithm.
# TI_REAL = TI[0]
# TI_COMPLEX = TI[1] + 1j * TI[2]
# TI_REAL = TI[0]
# TI_COMPLEX = TI[1::2] + 1j * TI[2::2]

# # Interpolator coefficients.
# P = np.array([
#     [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
#     [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
#     [1/3, -8/3, 10/3]])
# print(f"P:\n{P}")

NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.
# NEWTON_MAXITER = 10  # Maximum number of Newton iterations.
# MIN_FACTOR = 0.01  # Minimum allowed decrease in a step size.
# MAX_FACTOR = 1 / MIN_FACTOR  # Maximum allowed increase in a step size.


def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu):
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
    Z0 : ndarray, shape (3, n)
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
    n = y.shape[0]
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
        # U = h * np.linalg.inv(Mus) @ TI @ F
        f_real = -U[0]
        # f_real = -(h / MU_REAL) * U[0]
        # f_complex = -(U[1] + 1j * U[2])
        f_complex = np.empty((ncs, n), dtype=MU_COMPLEX.dtype)
        for i in range(ncs):
            f_complex[i] = -(U[2 * i + 1] + 1j * U[2 * i + 2])
            # f_complex[i] = -(h / MU_COMPLEX[i]) * (U[2 * i + 1] + 1j * U[2 * i + 2])

        dW_real = solve_lu(LU_real, f_real)
        # dW_complex = solve_lu(LU_complex, f_complex)
        dW_complex = np.empty_like(f_complex)
        for i in range(ncs):
            dW_complex[i] = solve_lu(LU_complex[i], f_complex[i])

        dW[0] = dW_real
        # dW[1] = dW_complex.real
        # dW[2] = dW_complex.imag
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


# TODO:
# - adapt documentation
# - fix error estimate of Fabien2009
# - dense output for yp
class RadauDAE(DaeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [4]_, where most of the ideas come from [2]_. 
    The embedded formula of [3]_ is applied to implicit differential equations.
    The error is controlled with a third-order accurate embedded formula as 
    introduced in [2]_ and refined in [3]_. The procedure is slightly adapted 
    by [4]_ to cope with implicit differential equations. A cubic polynomial 
    which satisfies the collocation conditions is used for the dense output.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below). The
        vectorized implementation allows a faster approximation of the Jacobian
        by finite differences (required for this solver).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
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
    jac_sparsity : {None, array_like, sparse matrix}, optional
        Defines a sparsity structure of the Jacobian matrix for a
        finite-difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few non-zero
        elements in *each* row, providing the sparsity structure will greatly
        speed up the computations [2]_. A zero entry means that a corresponding
        element in the Jacobian is always zero. If None (default), the Jacobian
        is assumed to be dense.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    mass_matrix : {None, array_like, sparse_matrix}, optional
        Defined the constant mass matrix of the system, with shape (n,n).
        It may be singular, thus defining a problem of the differential-
        algebraic type (DAE), see [1]. The default value is None.

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
    .. [5] N. Guglielmi, E. Hairer , "Implementing Radau IIA Methods for Stiff 
           Delay Differential Equations", Computing 67, 1-12, 2001.
    """
    def __init__(self, fun, t0, y0, yp0, t_bound, stages=3,
                 max_step=np.inf, rtol=1e-3, atol=1e-6, 
                 continuous_error_weight=0.5, jac=None, 
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

        self.h_abs_old = None
        self.error_norm_old = None

        self.stages = stages
        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
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
                # Z0 = self.sol(t + h * C).T - y
                Z0 = self.sol(t + h * C)[0].T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    # Fabien (5.59) and (5.60)
                    # TODO: Is it benefiction to multipy everything by
                    # (h * Lambda), hence MU_REAL * h * Jy instead of (1 / h)?
                    # => doesn't matter for van der pol and pendulum
                    LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                    LU_complex = [self.lu(MU / h * Jyp + Jy) for MU in MU_COMPLEX]
                    # LU_real = self.lu(Jyp + (h / MU_REAL) * Jy)
                    # LU_complex = [self.lu(Jyp + (h / MU) * Jy) for MU in MU_COMPLEX]

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU_real, LU_complex, self.solve_lu)

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp, f)
                    current_jac = True
                    LU_real = None
                    LU_complex = None

            if not converged:
                # print(f"not converged")
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
            # evaluate polynomial
            # TODO: Preevaluate the coefficients and 
            def eval_collocation_polynomial2(xi, C, Y):
                # # compute coefficients
                # V = np.vander(C, increasing=True)
                # # V = np.vander([0, *C], increasing=True)[1:, 1:].T
                # # V_inv = np.linalg.inv(V)
                # # coeffs = V_inv @ Y
                # coeffs = np.linalg.solve(V, Y)

                # # compute evaluation point
                # x = (xi - C[0]) / (C[-1] - C[0])

                # # add summands
                # return np.sum(
                #     [ci * x**i for i, ci in enumerate(coeffs)],
                #     axis=0,
                # )

                # Compute coefficients using Vandermonde matrix
                V = np.vander(C, increasing=True)
                coeffs = np.linalg.solve(V, Y)

                # Evaluate polynomial at xi
                n = len(C) - 1  # Degree of the polynomial
                # x = (xi - C[0]) / (C[-1] - C[0])
                x = xi
                powers_of_x = np.array([x**i for i in range(n + 1)])
                return np.dot(coeffs.T, powers_of_x)
            
            # Lagrange polynomial
            def eval_collocation_polynomial(xi, C, Y):
                s, m = Y.shape
                y = np.zeros(m)

                for i in range(s):
                    li = 1.0
                    for j in range(s):
                        if j != i:
                            li *= (xi - C[j]) / (C[i] - C[j])
                    y += li * Y[i]

                return y
            

            # Compute coefficients using Vandermonde matrix
            V = np.vander(C, increasing=True)
            V_inv = np.linalg.inv(V)
            coeffs = V_inv @ Y
            # coeffs = np.linalg.solve(V, Y)
            p0_2 = coeffs[0] # TODO: That is cheap if V_inv is already available :)
            p0_2 = V_inv[0] @ Y # and that is even cheaper ;)

            # p0_2 = eval_collocation_polynomial2(0.0, C, Y)
            # p0_2 = compute_interp(C, Y, 0.0)
            # # p0_2 = compute_interp(C, Y, 0.0, df=Yp)
            # p0_ = eval_collocation_polynomial2(0.0, C, Y)
            p0_ = eval_collocation_polynomial(0.0, C, Y)
            # p1_ = eval_collocation_polynomial(1, C, Y)
            # print(f"p0_: {p0_}")
            # print(f"p0_2: {p0_2}")
            # error_collocation = y - p0_
            error_collocation = y - p0_2

            # c1, c2, c3 = C
            # l1 = c2 * c3 / ((c1 - c2) * (c1 - c3))
            # l2 = c1 * c3 / ((c2 - c1) * (c2 - c3))
            # l3 = c1 * c2 / ((c3 - c1) * (c3 - c2))
            # error_collocation = y - l1 * Y[0] - l2 * Y[1] - l3 * Y[2]

            # # print(f"p0_2 - p0_: {p0_2 - p0_}")
            # assert np.allclose(p1_, Y[-1])
            # error_collocation = y - p0_
            # error = error_collocation
            # # error = y - p0_2
            # # print(f"p0_ collocation: {p0_}")
            # # print(f"error collocation: {error}")
            
            ###############
            # Fabien (5.65)
            ###############
            # # TODO: This is the embedded method that is stabilized below
            # error_Fabien = h * MU_REAL * (v @ Yp - b0 * yp - yp_new / MU_REAL)
            yp_hat_new = MU_REAL * (v @ Yp - b0 * yp)
            F = self.fun(t_new, y_new, yp_hat_new)
            # TODO: Is this already l-stable or do we have to add a possible second 
            # multiplication with LU_real?
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
            # # if True:
            #     # print(f"rejected")
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
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU_real = None
                LU_complex = None
                rejected = True
            else:
                step_accepted = True

        # Step is converged and accepted
        # TODO: Make this rate a user defined argument
        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
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
        Q = np.dot(self.Z.T, P)
        Yp = (A_inv / (self.h_abs * self.direction)) @ self.Z
        Qp = np.dot(Yp.T, P)
        # return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)
        return RadauDenseOutput(self.t_old, self.t, self.y_old, Q, self.yp_old, Qp)

    def _dense_output_impl(self):
        return self.sol

class RadauDenseOutput(DenseOutput):
    # def __init__(self, t_old, t, y_old, Q):
    def __init__(self, t_old, t, y_old, Q, yp_old, Qp):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.Qp = Qp
        self.order = Q.shape[1] - 1
        self.y_old = y_old
        self.yp_old = yp_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake.
        y = np.dot(self.Q, p)
        yp = np.dot(self.Qp, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
            yp += self.yp_old[:, None]
        else:
            y += self.y_old
            yp += self.yp_old

        return y, yp
