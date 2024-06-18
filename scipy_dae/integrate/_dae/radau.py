import numpy as np
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver
from .base import DAEDenseOutput as DenseOutput


S6 = 6 ** 0.5

# Butcher tableau. A is not used directly, see below.
C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
print(f"E:\n{E}")

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
# b_hat = b + (E * gamma) @ A
# (b_hat - b) @ A_inv = MU_REAL * E
# (b_hat - b) = MU_REAL * E @ A
b_hat = b + MU_REAL * E @ A

# de Swart1997
s = 3
vander = np.zeros((s, s))
for i in range(s):
    for j in range(s):
        vander[i, j] = C[j]**i
print(f"vander:\n{vander}")

rhs = 1 / np.arange(1, s + 1)
print(f"rhs:\n{rhs}")

# gamma0 = MU_REAL
gamma0 = 1 / MU_REAL
b0_hat = 0.02
b0_hat = gamma0 # That leads to v!!!!
# rhs[0] -= gamma0
rhs[0] -= b0_hat #+ gamma0
rhs -= gamma0
print(f"rhs:\n{rhs}")

b_hat = np.linalg.solve(vander, rhs)
print(f"b_hat:\n{b_hat}")

E = (b_hat - b) @ A_inv / MU_REAL
print(f"E:\n{E}")

# Interpolator coefficients.
P = np.array([
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])

NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


c1c2c3 = np.prod(C)
# TODO: How to derive this array: According to Fabien2009 below (5.65) v = b - b_hat.
# Consequently, his b_hat differs from that of Hairer.
v = np.array([0.428298294115369, -0.245039074384917, 0.366518439460903])
v2 = (b - b_hat)
print(f"v: {v}")
print(f"v2: {v2}")
# assert np.allclose(v, v2)
v = v2


from numpy.polynomial import Polynomial as Poly

def lagrange_basis_np(x_nodes, i):
    """ Construct the Lagrange basis polynomial L_i(x) using numpy.polynomial.Polynomial. """
    numerator = Poly([1])
    denominator = 1
    for j in range(len(x_nodes)):
        if j != i:
            numerator *= Poly([-x_nodes[j], 1])
            denominator *= (x_nodes[i] - x_nodes[j])
    return numerator / denominator

def hermite_basis_np(x_nodes):
    """ Construct the Hermite basis polynomials H_i(x) and K_i(x) using numpy.polynomial.Polynomial. """
    n = len(x_nodes)
    basis_funcs = []

    for i in range(n):
        Li = lagrange_basis_np(x_nodes, i)
        Li_prime = Li.deriv()

        # H_i(x) = (1 - 2 * (x - x_i) * L_i'(x_i)) * (L_i(x))^2
        Hi = (Poly([1]) - 2 * Poly([-x_nodes[i], 1]) * Li_prime(x_nodes[i])) * (Li**2)
        
        # K_i(x) = (x - x_i) * (L_i(x))^2
        Ki = Poly([-x_nodes[i], 1]) * (Li**2)
        
        basis_funcs.append((Hi, Ki))
    
    return basis_funcs


def eval_Hermite(t, C, Y, Yp):
    # from numpy.polynomial.hermite import hermvander, Hermite

    # n, m = Y.shape
    
    # # Construct the target vector (function values and derivatives)
    # b = np.zeros((2 * n, m))
    # b[0::2] = Y
    # b[1::2] = Yp

    # # # Construct the Hermite Vandermonde matrix
    # # V = hermvander(C_extended, 2 * n - 1)
    # # Construct the Hermite Vandermonde matrix
    # V = np.vander(C, increasing=True)
    # mult = np.arange(1, n + 1)
    # # mult[0] = 0
    # VV = np.block([
    #     [V, np.zeros_like(V)],
    #     [np.zeros_like(V), V * mult[None, :]],
    # ])

    # b = np.concatenate((Y, Yp))
    # # TODO: This gives two different polynomial coefficients???
    # coeffs = np.linalg.solve(VV, b)

    # from numpy.polynomial import Polynomial as Poly
    # ps = []
    # for coeff in coeffs.T:
    #     ps.append(
    #         Poly(coeff, domain=[0, 1], window=[0, 1])
    #     )

    # y = np.zeros_like(Y[0])
    # yp = np.zeros_like(Yp[0])
    # for i, p in enumerate(ps):
    #     y[i] = p(t)
    #     yp[i] = p.deriv()(t)

    # return y, yp

    y = np.zeros_like(Y[0])
    yp = np.zeros_like(Yp[0])
    basis = hermite_basis_np(C)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 1)
    # ts = np.linspace(0, 1, num=100)
    # for i, (Hi, Ki) in enumerate(basis):
    #     ax[0].plot(ts, Hi(ts), label=f"H{i}")
    #     ax[0].plot(C, np.ones_like(C), "ok", label="C")
    #     ax[1].plot(ts, Ki(ts), label=f"K{i}")
    #     # ax[1].plot(ts, Ki.deriv()(ts), label=f"K'{i}")
    #     ax[1].plot(C, np.ones_like(C), "ok", label="C")
    # ax[0].grid()
    # ax[0].legend()
    # ax[1].grid()
    # ax[1].legend()
    # plt.show()
    for i, (Hi, Ki) in enumerate(basis):
        y += Hi(t) * Y[i] + Ki(t) * Yp[i]
        yp += Hi.deriv()(t) * Y[i] + Ki.deriv()(t) * Yp[i]
    return y, yp

# Hermite and Lagrange interpolation using Vandermond matrix,
# see https://nicoguaro.github.io/index-2.html
def vander_mat(x):
    n = len(x)
    van = np.zeros((n, n))
    power = np.array(range(n))
    for row in range(n):
        van[row, :] = x[row]**power
    return van


def conf_vander_mat(x):
    n = len(x)
    conf_van = np.zeros((2*n, 2*n))
    power = np.array(range(2*n))
    for row in range(n):
        conf_van[row, :] = x[row]**power
        conf_van[row + n, :] = power*x[row]**(power - 1)
    return conf_van


def inter_coef(x, inter_type="lagrange"):
    if inter_type == "lagrange":
        vand_mat = vander_mat(x)
    elif inter_type == "hermite":
        vand_mat = conf_vander_mat(x)
    # coef = np.linalg.solve(vand_mat, np.eye(vand_mat.shape[0]))
    coef = np.linalg.inv(vand_mat)
    return coef


def compute_interp(x, f, x_eval, df=None):
    x_eval = np.atleast_1d(x_eval)
    n = len(x)
    if df is None:
        coef = inter_coef(x, inter_type="lagrange")
    else:
        coef = inter_coef(x, inter_type="hermite")
    f_eval = np.zeros((len(x_eval), f.shape[1]))
    nmat = coef.shape[0]
    for row in range(nmat):
        for col in range(nmat):
            if col < n or nmat == n:
                f_eval += coef[row, col]*x_eval**row*f[col]
            else:
                f_eval += coef[row, col]*x_eval**row*df[col - n]
    return f_eval


def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex, solve_lu, jac, lu):
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

    # W = V of Fabien
    # A_inv = W of Fabien
    Z = Z0
    W = TI.dot(Z0)
    Yp = (A_inv / h) @ Z
    Y = y + Z

    F = np.empty((3, n))
    tau = t + h * C

    dW_norm_old = None
    dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(3):
            F[i] = fun(tau[i], Y[i], Yp[i])

        if not np.all(np.isfinite(F)):
            break

        U = TI @ F
        f_real = -U[0]
        f_complex = -(U[1] + 1j * U[2])

        dW_real = solve_lu(LU_real, f_real)
        dW_complex = solve_lu(LU_complex, f_complex)

        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        # # solve collocation system without complex transformations
        # # Jy0, Jyp0 = jac(tau[0], Y[0], Yp[0], F[0])
        # # Jy1, Jyp1 = jac(tau[1], Y[1], Yp[1], F[1])
        # Jy2, Jyp2 = jac(tau[2], Y[2], Yp[2], F[2])
        # J = np.kron(np.eye(3), Jy2) + np.kron((A_inv / h), Jyp2)
        # # from scipy.linalg import block_diag
        # # J = block_diag([Jy0, Jy1, Jy2]) + 
        # LU = lu(J)
        # dZ = solve_lu(LU, -F.reshape(-1))
        # dW = TI.dot(dZ.reshape(3, -1))

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
    p = 3
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        # multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / p)

    with np.errstate(divide='ignore'):
        # factor = min(1, multiplier) * error_norm ** -0.25
        factor = min(1, multiplier) * error_norm ** (-1 / p)

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
                Z0 = np.zeros((3, y.shape[0]))
            else:
                # Z0 = self.sol(t + h * C).T - y
                Z0 = self.sol(t + h * C)[0].T - y

            scale = atol + np.abs(y) * rtol

            converged = False
            while not converged:
                if LU_real is None or LU_complex is None:
                    # Fabien (5.59) and (5.60)
                    LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                    LU_complex = self.lu(MU_COMPLEX / h * Jyp + Jy)

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU_real, LU_complex, self.solve_lu, self.jac, self.lu)

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
            # yp_new = ((A_inv / h) @ Z)[-1]
            y_new = Y[-1]
            yp_new = Yp[-1]

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol

            if True:
                # # ######################################################
                # # # error estimate by difference w.r.t. embedded formula
                # # ######################################################
                # # # compute embedded formula
                # # gamma0 = MU_REAL
                # # y_new_hat = y + h * gamma0 * yp + h * b_hat @ Yp

                # # # # # embedded trapezoidal step
                # # # # y_new_hat = y + 0.5 * h * (yp + Yp[-1])

                # # # y_new = y + h * (b @ Yp)
                # # error = y_new_hat - y_new
                # # # # error = Yp.T.dot(E) + h * gamma0 * yp
                # # # error = gamma0 * Yp.T.dot(E) + h * yp

                # # # # ZE = Z.T.dot(E) #/ h
                # # # # error = (yp + Jyp.dot(ZE)) * h
                # # # # error = (yp + Z.T @ E / h)
                # # # # error = (yp + Z.T @ E) * h
                # # # # error = Jyp @ (yp + Z.T @ E) * h
                # # # # error = (f + Jyp.dot(ZE)) #* (h / MU_REAL)
                # # # # error = (h * yp + Yp.T @ (b_hat - b)) / h
                # # # error = (yp + Yp.T @ (b_hat - b) / h)
                # # # error = self.solve_lu(LU_real, error)
                # # # # error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))

                # # ###########################
                # # # decomposed error estimate
                # # ###########################
                # # gamma0h = MU_REAL * h

                # # # scale E by MU_real since this is already done by Hairer's 
                # # # E that is used here
                # # e = E * MU_REAL

                # # # embedded thirs order method
                # # err = gamma0h * yp + Z.T.dot(e)

                # # # use bad error estimate
                # # error = err

                # ###########################
                # # decomposed error estimate
                # ###########################
                # # gamma0 = 1 / MU_REAL
                # gamma0 = MU_REAL

                # # scale E by MU_real since this is already done by Hairer's 
                # # E that is used here
                # e = E * gamma0

                # # embedded third order method
                # err = h * gamma0 * yp + Z.T.dot(e)
                # # err = h * MU_REAL * yp + Jyp @ Z.T.dot(E * MU_REAL)
                # # err = h * gamma0 * (yp - f) + Z.T.dot(e)

                # # embedded third order method, see Fabien2009 (5.65)
                # b0_hat = 0.02
                # # gamma_hat = 1 / MU_REAL
                # # yp_hat_new = (MU_REAL / h) * (y_new - (y + h * Z.T.dot(E * MU_REAL)))
                # # yp_hat_new = MU_REAL * (Z.T.dot(E * MU_REAL) / h - b0_hat * yp_new)
                # yp_hat_new = MU_REAL * (Yp.T.dot(E * MU_REAL) - b0_hat * yp)
                # # yp_hat_new = -(Z.T.dot(E * MU_REAL) / h - b0_hat * yp) / MU_REAL
                # err = -self.solve_lu(LU_real, self.fun(t_new, y_new, yp_hat_new))
                # # err = -self.solve_lu(LU_real, self.fun(t_new, y_new, yp_new))

                # # LU_real = self.lu(MU_REAL / h * Jyp + Jy)

                # # use bad error estimate
                # error = err




                # ####################
                # # ODE error estimate
                # ####################
                # err_ODE = h * MU_REAL * (yp - f) + Z.T.dot(E * MU_REAL)
                # LU_real_ODE = self.lu(MU_REAL / h * self.I - Jy)
                # error_ODE = self.solve_lu(LU_real_ODE, err_ODE) / (MU_REAL * h)
                # error = error_ODE

                # #####################################
                # # ODE with mass matrix error estimate
                # #####################################
                # # LU_real_ODE_mass = self.lu(MU_REAL / h * Jyp - Jy)
                # # # Fabien (5.59) and (5.60)
                # # LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                # # LU_real_ODE_mass = LU_real
                # # LU_real_ODE_mass = self.lu(MU_REAL / h * Jyp + Jy)
                # # err_ODE_mass = h * MU_REAL * Jyp @ ((yp - f) + Z.T.dot(E * MU_REAL))
                # # error_ODE_mass = self.solve_lu(LU_real_ODE_mass, err_ODE_mass) / (MU_REAL * h)
                # # error_ODE_mass = self.solve_lu(
                # #     LU_real_ODE_mass, 
                # #     Jyp @ ((yp - f) + Z.T.dot(E) / h),
                # # )
                # error_ODE_mass = self.solve_lu(
                #     LU_real, 
                #     # # (yp - f) + Jyp @ Z.T @ E / h,
                #     # yp + Jyp @ Z.T @ E / h,
                #     Jyp @ (yp + Z.T @ E / h), # TODO: Why is this a good estimate?
                # )
                # # error_ODE_mass = (yp + Z.T @ E / h) / (MU_REAL / h)
                # error = error_ODE_mass
                # # print(f"error_ODE_mass: {error_ODE_mass}")

                ############################################
                # error of collocation polynomial of order s
                ############################################                
                # evaluate polynomial
                # TODO: Why is this so bad conditioned?
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
                    x = (xi - C[0]) / (C[-1] - C[0])
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

                # # p0_2 = eval_collocation_polynomial2(0.0, C, Y)
                # p0_2 = compute_interp(C, Y, 0.0)
                # # p0_2 = compute_interp(C, Y, 0.0, df=Yp)
                p0_ = eval_collocation_polynomial(0.0, C, Y)
                p1_ = eval_collocation_polynomial(1, C, Y)
                # print(f"p0_2 - p0_: {p0_2 - p0_}")
                assert np.allclose(p1_, Y[-1])
                error_collocation = y - p0_
                error = error_collocation
                # error = y - p0_2
                # print(f"p0_ collocation: {p0_}")
                # print(f"error collocation: {error}")

                # ################
                # # Hermite spline
                # ################
                # Y_augmented = Y
                # # Y_augmented = np.concatenate((
                # #     # np.zeros_like(y)[None, :], Y,
                # #     # y[None, :], Y[:-1],
                # #     y[None, :], Y,
                # # ))
                # # Yp_augmented = b[:, None] * Yp
                # Yp_augmented = Yp
                # # Yp_augmented = np.concatenate((
                # #     # yp[None, :], Yp,
                # #     # yp[None, :], Yp[:-1],
                # #     np.zeros_like(yp)[None, :], Yp,
                # # ))
                # C_augmented = C
                # # C_augmented = np.array([0, *C])
                # # C_augmented = np.array([0, *C[:-1]])
                # p0_augmented, p0_p_augmented = eval_Hermite(0.0, C_augmented, Y_augmented, Yp_augmented)
                # p1_augmented, p1_p_augmented = eval_Hermite(C[0], C_augmented, Y_augmented, Yp_augmented)
                # p2_augmented, p2_p_augmented = eval_Hermite(C[1], C_augmented, Y_augmented, Yp_augmented)
                # p3_augmented, p3_p_augmented = eval_Hermite(C[2], C_augmented, Y_augmented, Yp_augmented)
                # error = y - p0_augmented
                # # error = yp - p0_p_augmented
                # # error[1:] = 0
                # # error = y_new - p1_augmented
                # # TODO: This is still wrong.
                # # assert np.allclose(p3_augmented, Y[-1])
                
                # # p0_ = eval_Hermite(0.0, C, Y, Yp)
                # # p1_ = eval_Hermite(1.0, C, Y, Yp)
                # # assert np.allclose(p1_, Y[-1])
                # # error = y - p0_
                # print(f"p0_ Hermite: {p0_augmented}")
                # print(f"y: {y}")

                # print(f"p1_ Hermite: {p1_augmented}")
                # print(f"Y1:          {Y[0]}")
                # print(f"p1_p Hermite: {p1_p_augmented}")
                # print(f"Yp1:          {Yp[0]}")

                # print(f"p2_ Hermite: {p2_augmented}")
                # print(f"Y2:          {Y[1]}")
                # print(f"p2_p Hermite: {p2_p_augmented}")
                # print(f"Yp2:          {Yp[1]}")

                # print(f"p3_ Hermite: {p3_augmented}")
                # print(f"Y3:          {Y[2]}")
                # print(f"p3_p Hermite: {p3_p_augmented}")
                # print(f"Yp3:          {Yp[2]}")

                # print(f"y_new: {y_new}")
                # print(f"error Hermite: {error}")
                # print(f"")
               
                # ###############
                # # Fabien (5.65)
                # ###############
                # b0_hat = 0.02
                # b0_hat = 1 / MU_REAL # TODO: I prefer this...
                # gamma_hat = 1 / MU_REAL
                # yp_hat_new = (y_new - (y + h * b_hat @ Yp + h * b0_hat * yp)) / (h * gamma_hat)
                # # yp_hat_new = (1 / gamma_hat) * ((b - b_hat) @ Yp - b0_hat * yp)
                # # yp_hat_new = (1 / gamma_hat) * ((b - b_hat) @ Yp - b0_hat * yp)
                # # yp_hat_new = MU_REAL * ((b - b_hat) @ Yp - b0_hat * yp)
                # # yp_hat_new = MU_REAL * ((b - b_hat) @ (A_inv / h) @ Z - b0_hat * yp)
                # # yp_hat_new = MU_REAL * (-MU_REAL * E @ Z / h - b0_hat * yp)
                # # yp_hat_new = (Z.T.dot(E) / h - yp)
                # F = self.fun(t_new, y_new, yp_hat_new)
                # Jy, Jyp = self.jac(t_new, y_new, yp_hat_new, F)
                # LU_real = self.lu(Jy + MU_REAL / h * Jyp)
                # error = self.solve_lu(LU_real, -F)
                # # error = self.solve_lu(LU_real, -F) / MU_REAL * h
                # print(f"t_new:          {t_new}")
                # print(f"yp_new:         {yp_new}")
                # print(f"yp_hat_new:     {yp_hat_new}")
                # # print(f"error_ODE_mass: {error_ODE_mass}")
                # print(f"error:          {error}")
                # print(f"")

                # error measure used in ride.m of Fabien2009
                # TODO: What is v?
                b0 = 1 / MU_REAL
                yp_hat_new = MU_REAL * (v @ Yp - b0 * yp)
                F = self.fun(t_new, y_new, yp_hat_new)
                error_Fabien = self.solve_lu(LU_real, -F)
                error = error_Fabien

                # # mix embedded error with collocation error as proposed in Guglielmi2001
                # error = 0.5 * (error_collocation + error_Fabien)
                
                error_norm = norm(error / scale)

                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

                if error_norm > 1:
                    factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                    h_abs *= max(MIN_FACTOR, safety * factor)

                    LU_real = None
                    LU_complex = None
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
