import numpy as np
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver


S6 = 6 ** 0.5

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3

# Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# and a complex conjugate pair. They are written below.
MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
              - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))

# These are transformation matrices.
T = np.array([
    [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
    [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
    [1, 1, 0]])
TI = np.array([
    [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
    [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
    [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# These linear combinations are used in the algorithm.
TI_REAL = TI[0]
TI_COMPLEX = TI[1] + 1j * TI[2]

# gamma = MU_REAL
# alpha = MU_COMPLEX.real
# beta = -MU_COMPLEX.imag
gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
# Lambda = np.array([
#     [gamma, 0, 0],
#     [0, alpha, -beta],
#     [0, beta, alpha],
# ])
Lambda = np.array([
    [gamma, 0, 0],
    [0, alpha, beta],
    [0, -beta, alpha],
])
denom = alpha**2 + beta**2
Lambda_inv = np.array([
    [denom / gamma, 0, 0],
    [0, alpha, -beta],
    [0, beta, alpha],
]) / denom

TLA = T @ Lambda
A_inv = T @ Lambda @ TI
A = T @ Lambda_inv @ TI
b = A[-1, :]
b_hat = b + (E * gamma) @ A

# print(f"gamma, alpha, beta: {[gamma, alpha, beta]}")
# print(f"A:\n{A}")
# print(f"np.linalg.inv(A):\n{np.linalg.inv(A)}")
# print(f"A_inv:\n{A_inv}")
# print(f"b:\n{b}")
# print(f"b_hat:\n{b_hat}")
# exit()

# Interpolator coefficients.
P = np.array([
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])

NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


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
    M_real = MU_REAL / h
    M_complex = MU_COMPLEX / h

    Z = Z0
    Yp = (1 / h) * A_inv @ Z
    Wp = TI.dot(Yp)

    # W = TI.dot(Z0)
    # Z = Z0

    F = np.empty((3, n))
    # ch = h * C
    tau = t + h * C

    # def F_composite(Yp):
    #     Yp = Yp.reshape(3, -1, order="C")
    #     Y = y + h * A @ Yp
    #     F = np.empty((3, n))
    #     for i in range(3):
    #         F[i] = fun(tau[i], Y[i], Yp[i])
    #     F = F.reshape(-1, order="C")
    #     return F

    # # def F_composite(W):
    # #     W = W.reshape(3, -1, order="C")
    # #     Z = T.dot(W)
    # #     Y = y + Z
    # #     Yp = (1 / h) * A_inv @ Z
    # #     F = np.empty((3, n))
    # #     for i in range(3):
    # #         F[i] = fun(tau[i], Y[i], Yp[i])
    # #     F = F.reshape(-1, order="C")
    # #     return F
    
    # from cardillo.math.fsolve import fsolve
    # from cardillo.solver import SolverOptions

    # Yp0 = Yp0.reshape(-1, order="C")
    # sol = fsolve(F_composite, Yp0, options=SolverOptions(numerical_jacobian_method="2-point"))
    # Yp = sol.x
    # Yp = Yp.reshape(3, -1, order="C")
    # Y = y + h * A @ Yp
    # Z = h * A @ Yp

    # # Z0 = Z0.reshape(-1, order="C")
    # # sol = fsolve(F_composite, Z0, options=SolverOptions(numerical_jacobian_method="2-point"))
    # # W = sol.x
    # # W = W.reshape(3, -1, order="C")
    # # Z = T.dot(W)
    # # Y = y + Z
    # # Yp = (1 / h) * A_inv @ Z

    # converged = sol.success
    # nit = sol.nit
    # rate = 1
    # return converged, nit, Y, Yp, Z, rate

    # dW_norm_old = None
    # dW = np.empty_like(W)
    dW_norm_old = None
    # dW = np.empty_like(W)
    dW = np.empty_like(Wp)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        # # Y = y + h * A @ T @ Wp
        # Y = y + A @ Zp
        # Yp = Zp / h
        # # Yp = (1 / h) * A_inv @ Zp
        # # Zp = h * A @ Yp

        # Z = T.dot(W)

        Yp = T.dot(Wp)
        Z = h * A @ Yp
        Y = y + h * A @ Yp
        # Y = y + Z

        # Y = y + Z
        # Yp = (1 / h) * A_inv @ Z
        for i in range(3):
            F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        # # f_real = F.T.dot(TI_REAL) - M_real * mass_matrix.dot(W[0])
        # # f_complex = F.T.dot(TI_COMPLEX) - M_complex * mass_matrix.dot(W[1] + 1j * W[2])
        # TODO: Both formulations are equivalend
        f_real = -M_real * F.T.dot(TI_REAL)
        f_complex = -M_complex * F.T.dot(TI_COMPLEX)
        # TIF = TI @ F
        # f_real = -M_real * TIF[0]
        # f_complex = -M_complex * (TIF[1] + 1j * TIF[2])

        # f_real = -MU_REAL * F.T.dot(TI_REAL)
        # f_complex = -MU_COMPLEX * F.T.dot(TI_COMPLEX)
        # # f_real = -h / MU_REAL * F.T.dot(TI_REAL)
        # # f_complex = -h / MU_COMPLEX * F.T.dot(TI_COMPLEX)

        dW_real = solve_lu(LU_real, f_real)
        dW_complex = solve_lu(LU_complex, f_complex)

        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        dW_norm = norm(dW / scale)
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        # print(F"rate: {rate}")
        if (rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
            break

        # W += dW
        # Z = T.dot(W)

        Wp += dW
        Yp = T.dot(Wp)
        Z = h * A @ Yp
        Y = y + h * A @ Yp
        # Y = y + Z

        if (dW_norm == 0 or rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        dW_norm_old = dW_norm

    # Y = y + Z
    # Yp = (1 / h) * A_inv @ Z

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
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
        # multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / (s + 1))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25
        # factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

    return factor


# TODO:
# - adapt documentation
class Radau(DaeSolver):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    The implementation follows [1]_. The error is controlled with a
    third-order accurate embedded formula. A cubic polynomial which satisfies
    the collocation conditions is used for the dense output.

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
    """
    def __init__(self, fun, t0, y0, yp0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, first_step, max_step, vectorized, jac, jac_sparsity)
        self.y_old = None

        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
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
        # print(f"t: {t}")

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

            # # TODO: 
            # # - Is there a better initial guess?
            # # - Do we iterate in y or yp?
            # # Yp0 = np.zeros((3, yp.shape[0]))
            # Yp0 = np.tile(yp[:, None], 3).T
            # Z0 = np.zeros((3, yp.shape[0]))
            # Z0 = np.tile(yp[:, None], 3).T
            if self.sol is None:
                Z0 = np.zeros((3, y.shape[0]))
            else:
                Z0 = self.sol(t + h * C).T - y

            # TODO: Which scale should we use?
            scale = atol + np.abs(y) * rtol
            # scale = atol + np.abs(yp) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                    LU_complex = self.lu(MU_COMPLEX / h * Jyp + Jy)
                    # LU_real = self.lu(h / MU_REAL * Jyp + Jy)
                    # LU_complex = self.lu(h / MU_COMPLEX * Jyp + Jy)

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
                h_abs *= 0.5
                LU_real = None
                LU_complex = None
                continue

            # Hairer1996 (8.2b)
            # y_new = y + Z[-1]
            y_new = Y[-1]
            yp_new = Yp[-1]
            # Z = Y - y
            # ZE = Z.T.dot(E) / h
            # error = self.solve_lu(LU_real, f + ZE)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # scale = atol + np.maximum(np.abs(yp), np.abs(yp_new)) * rtol

            if True:
                # compute embedded formula
                gamma0 = 1 / MU_REAL
                gamma0 = MU_REAL
                # b0_hat = MU_REAL
                y_new_hat = y + h * (gamma0 * yp + b_hat @ Yp)
                # y_new_hat = y + (h * gamma0 * yp + b_hat @ Z * h)

                # # embedded trapezoidal step
                # y_new_hat = y + 0.5 * h * (yp + Yp[-1])

                # y_new = y + h * (b @ Yp)
                error = y_new_hat - y_new
                # ZE = Z.T.dot(E) / h
                # error = ZE * h + gamma0 * yp * h
                    
                # ZE = Z.T.dot(E) / h
                # error = self.solve_lu(LU_real, f / h + ZE)

                # b0_hat = 0.02
                # # error = b0_hat / MU_REAL * (yp + Jyp.dot(ZE))
                # error = b0_hat / MU_REAL * (yp + ZE)

                # error = self.solve_lu(LU_real, f + Z.T.dot(E) / h)

                # print(f"error1: {error}")
                # error = h * (b0_hat * yp + (b_hat - b) @ Yp)
                # error = h * (1 * yp + (b_hat - b) @ Yp) / 1e1
                # # # error = b0_hat * yp + (b_hat - b) @ Yp
                # # # error *= h
                # # print(f"error2: {error}")

                error_norm = norm(error / scale)

                # # see [1], chapter IV.8, page 127
                # # TODO: Hairer1996, (8.20)
                # # TODO: Hairer1996, (8.19a)
                # # error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))
                # # de Swart1997 (15)
                # # b0 = MU_REAL => b_hat0 = b0 / MU_REAL = 1 => => b0 = b_hat0 * MU_REAL 
                # b0_hat = 0.02
                # # b0_hat = 0.02 * MU_REAL
                # # b0_hat = 0.02 / MU_REAL
                # # b0_hat = 1
                # # error = b0_hat / MU_REAL * (f + self.mass_matrix.dot(ZE))
                # error = self.solve_lu(LU_real, b0_hat * (f + self.mass_matrix.dot(ZE)))
                # error_norm = norm(error / scale)

                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

                # if rejected and error_norm > 1: # try with stabilised error estimate
                #     # error = self.solve_lu(LU_real, self.fun(t, y + error) + self.mass_matrix.dot(ZE))
                #     error = self.solve_lu(LU_real, b0_hat * (self.fun(t, y + error) + self.mass_matrix.dot(ZE)))
                #     # error = self.solve_lu(LU_real, self.fun(t, y + error, h) + self.mass_matrix.dot(ZE))
                #     error_norm = norm(error / scale)

                if error_norm > 1:
                    factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                    h_abs *= max(MIN_FACTOR, safety * factor)

                    LU_real = None
                    LU_complex = None
                    rejected = True
                else:
                    step_accepted = True
            else:
                step_accepted = True

        if True:
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

        f_new = self.fun(t_new, y_new, yp_new)

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
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        # Here we don't multiply by h, not a mistake.
        y = np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y
