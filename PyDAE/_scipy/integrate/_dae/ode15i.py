import numpy as np
from scipy.integrate._ivp.common import norm, EPS, warn_extraneous
from scipy.integrate._ivp.base import DenseOutput
from .dae import DaeSolver


# S6 = 6 ** 0.5

# # Butcher tableau. A is not used directly, see below.
# C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
# E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3

# # Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
# # and a complex conjugate pair. They are written below.
# MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
#               - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))

# # These are transformation matrices.
# T = np.array([
#     [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
#     [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
#     [1, 1, 0]])
# TI = np.array([
#     [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
#     [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
#     [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])

# # These linear combinations are used in the algorithm.
# TI_REAL = TI[0]
# TI_COMPLEX = TI[1] + 1j * TI[2]

# # gamma = MU_REAL
# # alpha = MU_COMPLEX.real
# # beta = -MU_COMPLEX.imag
# gamma = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
# alpha = 3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
# beta = 0.5 * (3 ** (5 / 6) + 3 ** (7 / 6))
# # Lambda = np.array([
# #     [gamma, 0, 0],
# #     [0, alpha, -beta],
# #     [0, beta, alpha],
# # ])
# Lambda = np.array([
#     [gamma, 0, 0],
#     [0, alpha, beta],
#     [0, -beta, alpha],
# ])
# denom = alpha**2 + beta**2
# Lambda_inv = np.array([
#     [denom / gamma, 0, 0],
#     [0, alpha, -beta],
#     [0, beta, alpha],
# ]) / denom

# TLA = T @ Lambda
# A_inv = T @ Lambda @ TI
# A = T @ Lambda_inv @ TI
# b = A[-1, :]
# b_hat = b + (E * gamma) @ A

# # print(f"gamma, alpha, beta: {[gamma, alpha, beta]}")
# # print(f"A:\n{A}")
# # print(f"np.linalg.inv(A):\n{np.linalg.inv(A)}")
# # print(f"A_inv:\n{A_inv}")
# # print(f"b:\n{b}")
# # print(f"b_hat:\n{b_hat}")
# # exit()

# # Interpolator coefficients.
# P = np.array([
#     [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
#     [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
#     [1/3, -8/3, 10/3]])

MAX_ORDER = 5
NEWTON_MAXITER = 4  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

# Initialize method parameters for BDFs in Lagrangian form
# and constant step size.  Column k corresponds to the formula 
# of order k.  lcf holds the leading coefficient, cf holds 
# the rest.
lcf = np.array([1, 3/2, 11/6, 25/12, 137/60])
cf = np.array([
    [-1,  -2,   -3,   -4,    -5],
    [ 0, 1/2,  3/2,    3,     5],
    [ 0,   0, -1/3, -4/3, -10/3],
    [ 0,   0,    0,  1/4,   5/4],
    [ 0,   0,    0,    0,  -1/5],
])
     
# derM[:, k] contains coefficients for calculating scaled
# derivative of order k using equally spaced mesh.       
derM = np.array([
    [ 1,  1,  1,  1,   1,   1],
    [-1, -2, -3, -4,  -5,  -6],
    [ 0,  1,  3,  6,  10,  15],
    [ 0,  0, -1, -4, -10, -20],
    [ 0,  0,  0,  1,   5,  15],
    [ 0,  0,  0,  0,  -1,  -6],
    [ 0,  0,  0,  0,   0,   1],
])


def weights(x, xi, maxder):
    """
    Compute Lagrangian interpolation coefficients c for the value at xi 
    of a polynomial interpolating at distinct nodes x(1),...,x(N) and
    derivatives of the polynomial of orders 0,...,maxder.  c[j,d+1] 
    is the coefficient of the function value corresponding to x[j] when
    computing the derivative of order d.  Note that maxder <= N-1.
    
    This program is based on the Fortran code WEIGHTS1 of B. Fornberg, 
    A Practical Guide to Pseudospectral Methods, Cambridge University
    Press, 1995.
    """
    
    n = len(x) - 1
    c = np.zeros((n + 1, maxder + 1))
    c[0, 0] = 1
    tmp1 = 1
    tmp4 = x[0] - xi
    
    for i in range(1, n + 1):
        mn = min(i, maxder)
        tmp2 = 1
        tmp5 = tmp4
        tmp4 = x[i] - xi
        for j in range(i):
            tmp3 = x[i] - x[j]
            tmp2 *= tmp3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = tmp1 * (k * c[i-1, k-1] - tmp5 * c[i-1, k]) / tmp2
                c[i, 0] = -tmp1 * tmp5 * c[i-1, 0] / tmp2
            for k in range(mn, 0, -1):
                c[j, k] = (tmp4 * c[j, k] - k * c[j, k-1]) / tmp3
            c[j, 0] = tmp4 * c[j, 0] / tmp3
        tmp1 = tmp2
    
    return c


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
    if unknown_z:
        W = TI.dot(Z0)
        Yp = A_inv @ Z / h
    else:
        Yp = A_inv @ Z
        if unknown_densities:
            Yp /= h
        W = TI.dot(Yp)

    F = np.empty((3, n))
    tau = t + h * C

    if False:
        if unknown_z:
            def F_composite(Z):
                Z = Z.reshape(3, -1, order="C")
                Yp = A_inv @ Z / h
                Y = y + Z
                F = np.empty((3, n))
                for i in range(3):
                    F[i] = fun(tau[i], Y[i], Yp[i])
                F = F.reshape(-1, order="C")
                return F
        else:
            def F_composite(Yp):
                Yp = Yp.reshape(3, -1, order="C")
                Z = A @ Yp
                if unknown_densities:
                    Z *= h
                Y = y + Z
                F = np.empty((3, n))
                for i in range(3):
                    if unknown_densities:
                        F[i] = fun(tau[i], Y[i], Yp[i])
                    else:
                        F[i] = fun(tau[i], Y[i], Yp[i] / h)
                F = F.reshape(-1, order="C")
                return F
        
        from cardillo.math.fsolve import fsolve
        from cardillo.solver import SolverOptions

        if unknown_z:
            Z = Z.reshape(-1, order="C")
            sol = fsolve(F_composite, Z, options=SolverOptions(numerical_jacobian_method="2-point", newton_max_iter=NEWTON_MAXITER))
            Z = sol.x
            Z = Z.reshape(3, -1, order="C")
            Yp = A_inv @ Z / h
        else:
            Yp = Yp.reshape(-1, order="C")
            sol = fsolve(F_composite, Yp, options=SolverOptions(numerical_jacobian_method="2-point", newton_max_iter=NEWTON_MAXITER))
            Yp = sol.x
            Yp = Yp.reshape(3, -1, order="C")
            Z = A @ Yp
            if unknown_densities:
                Z *= h

        Y = y + Z
        
        converged = sol.success
        nit = sol.nit
        rate = 1
        return converged, nit, Y, Yp, Z, rate

    else:
        dW_norm_old = None
        dW = np.empty_like(W)
        converged = False
        rate = None
        for k in range(NEWTON_MAXITER):
            if unknown_z:
                Z = T.dot(W)
                Yp = A_inv @ Z / h
            else:
                Yp = T.dot(W)
                Z = A @ Yp
                if unknown_densities:
                    Z *= h
            Y = y + Z
            for i in range(3):
                if unknown_densities:
                    F[i] = fun(tau[i], Y[i], Yp[i])
                else:
                    F[i] = fun(tau[i], Y[i], Yp[i] / h)

            if not np.all(np.isfinite(F)):
                break

            # f_real = F.T.dot(TI_REAL) - M_real * mass_matrix.dot(W[0])
            # f_complex = F.T.dot(TI_COMPLEX) - M_complex * mass_matrix.dot(W[1] + 1j * W[2])

            if unknown_z:
                f_real = -h / MU_REAL * F.T.dot(TI_REAL)
                f_complex = -h / MU_COMPLEX * F.T.dot(TI_COMPLEX)
                # TIF = TI @ F
                # f_real = -h / MU_REAL * TIF[0]
                # f_complex = -h / MU_COMPLEX * (TIF[1] + 1j * TIF[2])

                # P = np.kron(TI, np.eye(n))
                # QI = np.kron(T, np.eye(n))
                # B = np.kron(h * Lambda, J)
                # np.set_printoptions(3, suppress=True)
                # print(f"P:\n{P}")
                # print(f"QI:\n{QI}")
                # J = P @ 
            else:
                if unknown_densities:
                    # TODO: Both formulations are equivalend
                    f_real = -M_real * F.T.dot(TI_REAL)
                    f_complex = -M_complex * F.T.dot(TI_COMPLEX)
                    # TIF = TI @ F
                    # f_real = -M_real * TIF[0]
                    # f_complex = -M_complex * (TIF[1] + 1j * TIF[2])
                else:
                    f_real = -MU_REAL * F.T.dot(TI_REAL)
                    f_complex = -MU_COMPLEX * F.T.dot(TI_COMPLEX)

            dW_real = solve_lu(LU_real, f_real)
            dW_complex = solve_lu(LU_complex, f_complex)

            dW[0] = dW_real
            dW[1] = dW_complex.real
            dW[2] = dW_complex.imag

            dW_norm = norm(dW / scale)
            if dW_norm_old is not None:
                rate = dW_norm / dW_norm_old

            # print(F"dW_norm: {dW_norm}")
            # print(F"rate: {rate}")
            # if rate is not None:
            #     print(F"rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm: {rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm}")
            # print(F"tol: {tol}")
            if (rate is not None and (rate >= 1 or rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
                break

            W += dW
            if unknown_z:
                Z = T.dot(W)
                Yp = A_inv @ Z / h
            else:
                Yp = T.dot(W)
                Z = A @ Yp
                if unknown_densities:
                    Z *= h
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
    # s = 3
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
        # multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / (s + 1))

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25
        # factor = min(1, multiplier) * error_norm ** (-1 / (s + 1))

    # print(f"factor: {factor}")

    return factor



class ODE15I(DaeSolver):
    def __init__(self, fun, t0, y0, yp0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, yp0, t_bound, rtol, atol, first_step, max_step, vectorized, jac, jac_sparsity)
        # self.y_old = None
        self.order_old = None
        self.h_abs_old = None
        # self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None

        self.current_jac = True
        self.LU = None
        self.Z = None

        self.mesh = np.zeros(MAX_ORDER + 2)
        self.mesh[0] = self.t

        self.meshsol = np.zeros((self.n, MAX_ORDER + 2))
        self.meshsol[:, 0] = self.y

        # Using the initial slope, create fictitious solution at t - h for 
        # starting the integration.
        h = self.h_abs * self.direction
        self.mesh[1] = self.t - h
        self.meshsol[:, 1] = self.y - h * self.yp

        # TODO: What does this mean?
        self.nconh = 1
        self.order = 1
        self.PDscurrent = True # TODO: I think this is current_jac

    def _step_impl(self):
        t = self.t
        y = self.y
        yp = self.yp
        f = self.f
        print(f"t: {t}")

        max_step = self.max_step
        atol = self.atol
        rtol = self.rtol
        threshold = atol / rtol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            h_abs_old = None
            # error_norm_old = None
        elif self.h_abs < min_step:
            h_abs = min_step
            h_abs_old = None
            # error_norm_old = None
        else:
            h_abs = self.h_abs
            h_abs_old = self.h_abs_old
            # error_norm_old = self.error_norm_old

        PDscurrent = self.PDscurrent
        order = self.order
        order_old = self.order_old
        mesh = self.mesh
        meshsol = self.meshsol

        Jy = self.Jy
        Jyp = self.Jyp
        LU = self.LU
        current_jac = self.current_jac
        jac = self.jac

        rejected = False
        step_accepted = False
        message = None
        gotynew = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            invwt = np.ones_like(y) / np.maximum(np.abs(y), threshold)

            if h_abs != h_abs_old or order != order_old:
                if h_abs != h_abs_old:
                    self.nconh = 0

                Miter = Jy + (lcf[order] / h) * Jyp
                LU = self.lu(Miter)
                havrate = False
                rate = 1.0 # dummy value for test

            # Predict the solution and its derivative at t_new.
            c = weights(mesh[:order + 1], t_new, 1)
            ynew = meshsol[:, :order + 1] @ c[:, 0]
            ypnew = meshsol[:, :order + 1] @ c[:, 1]
            ypred = ynew
            minnrm = 100 * EPS * np.linalg.norm(ypred * invwt, np.inf)

            # Compute local truncation error constant.
            erconst = -1 / (order + 1)
            for j in range(1, order):
                prod = cf[j, order] * np.prod(
                    ((t - (j - 1) * h) - mesh[:order + 1]) / (h * np.arange(1, order + 2))
                )
                erconst = erconst- cf[j, order] * prod
            erconst = abs(erconst)

            # Iterate with simplified Newton method.
            # TODO: Move this into top level function
            tooslow = False
            for iter in range(NEWTON_MAXITER):
                rhs = -self.fun(t_new, ynew, ypnew)
                del_ = self.solve_lu(LU, rhs)
                
                newnrm = np.linalg.norm(del_ * invwt, np.inf)
                ynew = ynew + del_
                ypnew = ypnew + (lcf[order] / h) * del_
                
                if iter == 0:
                    if newnrm <= minnrm:
                        gotynew = True
                        break
                    savnrm = newnrm
                else:
                    rate = (newnrm / savnrm) ** (1 / iter)
                    havrate = True
                    if rate > 0.9:
                        tooslow = True
                        break
                
                if havrate and (newnrm * rate / (1 - rate) <= 0.33 * rtol):
                    gotynew = True
                    break
                elif iter == NEWTON_MAXITER - 1:
                    tooslow = True
                    break

            if tooslow:
                h_abs_old = h_abs
                order_old = order
                # Speed up the iteration by forming new linearization or reducing h.
                if not PDscurrent:
                    f_new = self.fun(t_new, y, yp)
                    Jy, Jyp = self.jac(t_new, y_new, yp_new, f_new)
                    PDscurrent = True
                              
                    # Set a dummy value of order_old to force formation of iteration matrix.
                    order_old = 0
            else:
                h_abs = 0.25 * h_abs
                done = False

            # Using the tentative solution, approximate scaled derivative 
            # used to estimate the error of the step.
            norm_term = np.linalg.norm((ynew - ypred) * invwt, np.inf)
            numerator = h_abs * np.arange(1, order + 2)
            denominator = t_new - mesh[:order + 1]
            sderkp1 = np.linalg.norm((ynew - ypred) * invwt, np.inf) * np.abs(np.prod(numerator / denominator))
            erropt = sderkp1 / (order + 1);  # Error assuming constant step size.    
            err = sderkp1 * erconst;         # Error accounting for irregular mesh.


            # Approximate directly derivatives needed to consider lowering the
            # order.  Multiply by a power of h to get scaled derivative.
            order = order
            order_opt = order

            if order > 1:
                if self.nconh >= order:
                    sderk = np.linalg.norm((np.dot(np.column_stack([ynew, meshsol[:, :order]]), derM[:order+1, order]) * invwt), np.inf)
                else:
                    c = weights(np.append(t_new, mesh[:order]), t_new, order)
                    sderk = np.linalg.norm((np.dot(np.column_stack([ynew, meshsol[:, :order]]), c[:, order]) * invwt), np.inf) * h_abs**order

                if order == 2:
                    if sderk <= 0.5 * sderkp1:
                        order_opt = order - 1
                        erropt = sderk / order
                else:
                    if self.nconh >= order - 1:
                        sderkm1 = np.linalg.norm((np.dot(np.column_stack([ynew, meshsol[:, :order-1]]), derM[:order, order-1]) * invwt), np.inf)
                    else:
                        c = weights(np.append(t_new, mesh[:order-1]), t_new, order-1)
                        sderkm1 = np.linalg.norm((np.dot(np.column_stack([ynew, meshsol[:, :order-1]]), c[:, order-1]) * invwt), np.inf) * h_abs**(order-1)

                    if max(sderkm1, sderk) <= sderkp1:
                        order_opt = order - 1
                        erropt = sderk / order

            if err > rtol:
                if h_abs < min_step:
                # if h_abs <= min_step:
                    return False, self.TOO_SMALL_STEP
            else:
                step_accepted = True
                return step_accepted, message

            # h_abs_old = h_abs
            # order_old = order
            # nfails = nfails + 1
            # if nfails == 1:
            #     absh = absh * min(0.9,max(0.25, 0.9*(0.5*rtol/erropt)^(1/(kopt+1)))); 
            # case 2
            # absh = absh * 0.25;
            # otherwise
            # kopt = 1;
            # absh = absh * 0.25;
            # end
            # absh = max(absh,hmin);
            # if absh < abshlast
            # done = false;
            # end
            # k = kopt; 
             
            h_abs_old = h_abs
            order_old = order
            nfails += 1

            if nfails == 1:
                h_abs = h_abs * min(0.9, max(0.25, 0.9 * (0.5 * rtol / erropt)**(1 / (order_opt + 1))))
            elif nfails == 2:
                h_abs = h_abs * 0.25
            else:
                order_opt = 1
                h_abs = h_abs * 0.25

            h_abs = max(h_abs, min_step)
            done = not h_abs < h_abs_old
            order = order_opt
            
            print(f"")

            # for j = 2:k
            # erconst = erconst - ...
            #         cf(j,k)*prod(((t - (j-1)*h) - mesh(1:k+1)) ./ (h * (1:k+1)));
            # end
            # erconst = abs(erconst);       

            # if self.sol is None:
            #     Z0 = np.zeros((3, y.shape[0]))
            # else:
            #     Z0 = self.sol(t + h * C).T - y

            # scale = atol + np.abs(y) * rtol

            exit()

            converged = False
            while not converged:
                if LU is None or LU_complex is None:
                    if unknown_z:
                        # LU_real = self.lu(h / MU_REAL * Jyp + Jy)
                        # LU_complex = self.lu(h / MU_COMPLEX * Jyp + Jy)
                        LU = self.lu(Jyp + h / MU_REAL * Jy)
                        LU_complex = self.lu(Jyp + h / MU_COMPLEX * Jy)
                        # # TODO: This is only used for exact newton and the error estimate...
                        # LU_real = self.lu(MU_REAL / h * Jyp + Jy)
                        # LU_complex = self.lu(MU_COMPLEX / h * Jyp + Jy)
                    else:
                        LU = self.lu(MU_REAL / h * Jyp + Jy)
                        LU_complex = self.lu(MU_COMPLEX / h * Jyp + Jy)

                converged, n_iter, Y, Yp, Z, rate = solve_collocation_system(
                    self.fun, t, y, h, Z0, scale, self.newton_tol,
                    LU, LU_complex, self.solve_lu)
                # print(f"converged: {converged}")

                if not converged:
                    if current_jac:
                        break

                    Jy, Jyp = self.jac(t, y, yp, f)
                    current_jac = True
                    LU = None
                    LU_complex = None

            if not converged:
                h_abs *= 0.5
                # print(f"not converged")
                # print(f"h_abs: {h_abs}")
                LU = None
                LU_complex = None
                continue

            # Hairer1996 (8.2b)
            y_new = y + Z[-1]
            # y_new = Y[-1]
            yp_new = Yp[-1]
            if not unknown_densities:
                yp_new /= h

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # scale = atol + np.maximum(np.abs(yp), np.abs(yp_new)) * rtol
            # scale = atol + h * np.maximum(np.abs(yp), np.abs(yp_new)) * rtol

            if True:
                # ######################################################
                # # error estimate by difference w.r.t. embedded formula
                # ######################################################
                # # compute embedded formula
                # gamma0 = MU_REAL
                # y_new_hat = y + h * gamma0 * yp + h * b_hat @ Yp

                # # # # embedded trapezoidal step
                # # # y_new_hat = y + 0.5 * h * (yp + Yp[-1])

                # # y_new = y + h * (b @ Yp)
                # error = y_new_hat - y_new
                # # # error = Yp.T.dot(E) + h * gamma0 * yp
                # # error = gamma0 * Yp.T.dot(E) + h * yp

                # # # ZE = Z.T.dot(E) #/ h
                # # # error = (yp + Jyp.dot(ZE)) * h
                # # # error = (yp + Z.T @ E / h)
                # # # error = (yp + Z.T @ E) * h
                # # # error = Jyp @ (yp + Z.T @ E) * h
                # # # error = (f + Jyp.dot(ZE)) #* (h / MU_REAL)
                # # # error = (h * yp + Yp.T @ (b_hat - b)) / h
                # # error = (yp + Yp.T @ (b_hat - b) / h)
                # # error = self.solve_lu(LU_real, error)
                # # # error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))

                # ###########################
                # # decomposed error estimate
                # ###########################
                # gamma0h = MU_REAL * h

                # # scale E by MU_real since this is already done by Hairer's 
                # # E that is used here
                # e = E * MU_REAL

                # # embedded thirs order method
                # err = gamma0h * yp + Z.T.dot(e)

                # # use bad error estimate
                # error = err

                ###########################
                # decomposed error estimate
                ###########################
                gamma0 = 1 / MU_REAL

                # scale E by MU_real since this is already done by Hairer's 
                # E that is used here
                e = E * gamma0

                # embedded thirs order method
                err = h * gamma0 * yp + Z.T.dot(e)
                # err = h * gamma0 * (yp - f) + Z.T.dot(e)

                # use bad error estimate
                error = err

                # improve error estimate for stiff components
                if unknown_z:
                    # error = self.solve_lu(LU_real, (yp - f) + Z.T.dot(e) / (h * gamma0))
                    error = self.solve_lu(LU, err)
                    # print(f"err: {err}")
                    # print(f"error: {error}")
                    # print(f"h: {h}")
                    # error = self.solve_lu(LU_real, err / (h * gamma0))
                    # error = self.solve_lu(LU_real, (h * gamma0 * yp + Jyp @ Z.T.dot(e)))
                    # error = self.solve_lu(LU_real, err / (h * gamma0))
                    # error = self.solve_lu(LU_real, (yp + Z.T.dot(e) / (h * gamma0)))

                    # D = np.eye(self.n) / (h * gamma0) + Jy
                    # error = np.linalg.solve(D, err / (h * gamma0))
                    pass
                else:
                    # error = self.solve_lu(LU_real, err / (h * gamma0))
                    pass
                    # error = self.solve_lu(LU_real, err / gamma0h)
                    # error = self.solve_lu(LU_real, err * gamma0h)
                    # error = self.solve_lu(LU_real, (gamma0h * yp + Z.T.dot(e)) / gamma0h)
                    # error = self.solve_lu(LU_real, yp + Z.T.dot(e) / gamma0h)
                    # # error = self.solve_lu(LU_real, yp + Z.T.dot(e) / gamma0h)
                    # # error = self.solve_lu(LU_real, yp + Z.T.dot(E) / h)
                    # error = self.solve_lu(LU_real, yp + Z.T.dot(E) / h)
                    # error = self.solve_lu(LU_real, yp + Jyp @ Z.T.dot(E) / h)
                
                error_norm = norm(error / scale)

                safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

                if rejected and error_norm > 1: # try with stabilised error estimate
                    print(f"rejected")
                    # error = self.solve_lu(LU_real, error)
                    # # # error = self.solve_lu(LU_real, self.fun(t, y + error) + self.mass_matrix.dot(ZE))
                    # # error = self.solve_lu(LU_real, b0_hat * (self.fun(t, y + error) + self.mass_matrix.dot(ZE)))
                    # # # error = self.solve_lu(LU_real, self.fun(t, y + error, h) + self.mass_matrix.dot(ZE))
                    # error_norm = norm(error / scale)

                if error_norm > 1:
                    factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                    # print(f"h_abs: {h_abs}")
                    # print(f"factor: {factor}")
                    h_abs *= max(MIN_FACTOR, safety * factor)
                    # print(f"h_abs: {h_abs}")

                    LU = None
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
            # print(f"factor: {factor}")

            if not recompute_jac and factor < 1.2:
                factor = 1
            else:
                LU = None
                LU_complex = None

            f_new = self.fun(t_new, y_new, yp_new)
            if recompute_jac:
                Jy, Jyp = self.jac(t_new, y_new, yp_new, f_new)
                current_jac = True
            elif jac is not None:
                current_jac = False

            self.h_abs_old = self.h_abs
            self.error_norm_old = error_norm

            # print(f"h_abs: {h_abs}")
            self.h_abs = h_abs * factor
            # print(f"self.h_abs: {self.h_abs}")

        f_new = self.fun(t_new, y_new, yp_new)

        self.y_old = y
        self.yp_old = yp

        self.t = t_new
        self.y = y_new
        self.yp = yp_new
        self.f = f_new

        self.Z = Z

        self.LU_real = LU
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
