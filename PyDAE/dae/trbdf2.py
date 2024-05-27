import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (
    validate_max_step, validate_tol, select_initial_step,
    norm, EPS, num_jac, warn_extraneous, validate_first_step,
)


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

# Parameters of the method
# Butcher Tableau Coefficients of the method
gm = 2 - np.sqrt(2)
d = gm / 2
w = np.sqrt(2) / 4

# Buthcer Tableau Coefficients for the error estimator
e0 = (1 - w) / 3
e1 = w + (1 / 3)
e2 = d / 3

# Coefficients required for interpolating y'
pd0 = 1.5 + np.sqrt(2)
pd1 = 2.5 + 2 * np.sqrt(2)
pd2 = -(6 + 4.5 * np.sqrt(2))


def solve_trbdf2_system(fun, z0, scale, tol, LU, solve_lu, mass_matrix):
    z = z0.copy() # why is this mandatory?
    dz_norm_old = None
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        f = fun(z)
        if not np.all(np.isfinite(f)):
            break

        dz = solve_lu(LU, f)
        dz_norm = norm(dz / scale)
        if dz_norm_old is not None:
            rate = dz_norm / dz_norm_old

        if (rate is not None and (rate >= 1 or
                                  rate ** (NEWTON_MAXITER - k) / (1 - rate) * dz_norm > tol)):
            break

        z += dz

        if (dz_norm == 0 or
                rate is not None and rate / (1 - rate) * dz_norm < tol):
            converged = True
            break

        dz_norm_old = dz_norm

    return converged, k + 1, z, rate


# TODO: Which exponent should we use?
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
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** (1 / 2)

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** (-1 / 2)

    return factor


class TRBDF2(OdeSolver):
    """
    The TR-BDF2 method was originally proposed in [1]_ as a composite method consisting of
    one stage of the Trapezoidal method followed by a stage of the 2nd order BDF method.
    The stages are setup in such way that both of them have the same Jacobian matrix.
    In [2]_ this method has been shown to be equivalent to an ESDIRK method of order 2.
    The TR-BDF2 method has been a popular choice for applications such as circuit simulations
    and weather simulation. This owes to the fact that it has several properties which make it
    an "optimal" method among all 3-stage DIRK methods. Some of these are as follows:

    1.) It is first same as last (FSAL); only two implicit stages need to be evaluated at every step.
    2.) The Jacobian matrix for both stages is the same.
    3.) It has an embedded 3rd order method for error estimation.
    4.) The method is strongly s-stable, making it ideal for stiff problems.
    5.) It is endowed with a cubic interpolant for dense output.

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
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits). But if a component of `y`
        is approximately below `atol`, the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
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
        speed up the computations [3]_. A zero entry means that a corresponding
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
    .. [1] Bank, Randolph E., et al. "Transient simulation of silicon devices and circuits."
           IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 4.4 (1985): 436-451.
    .. [2] Hosea, M. E., and L. F. Shampine. "Analysis and implementation of TR-BDF2."
           Applied Numerical Mathematics 20.1-2 (1996): 21-37.
    .. [3] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
           sparse Jacobian matrices", Journal of the Institute of Mathematics
           and its Applications, 13, pp. 117-120, 1974.
    """

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, jac=None, jac_sparsity=None,
                 vectorized=False, first_step=None, mass_matrix=None, 
                 var_index=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        # Select initial step assuming the same order which is used to control
        # the error
        if first_step is None:
            self.h_abs = select_initial_step(self.fun, self.t, self.y,
                                             self.f, self.direction,
                                             2, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.h_abs_old = None
        self.error_norm_old = None

        self.newton_tol = max(10 * EPS / rtol, min(0.03, rtol ** 0.5))
        self.sol = None
        self.jac_factor = None
        self.jac, self.J = self._validate_jac(jac, jac_sparsity)

        if issparse(self.J):
            def lu(A):
                self.nlu += 1
                return splu(A)

            def solve_lu(LU, b):
                return LU.solve(b)

            I = eye(self.n, format='csc')
        else:
            def lu(A):
                self.nlu += 1
                return lu_factor(A, overwrite_a=True)

            def solve_lu(LU, b):
                return lu_solve(LU, b, overwrite_b=True)

            I = np.identity(self.n)

        self.lu = lu
        self.solve_lu = solve_lu
        self.I = I
        
        self.mass_matrix, self.index_algebraic_vars, self.nvars_algebraic = \
                          self._validate_mass_matrix(mass_matrix)
        # TODO: Validate this
        self.var_index = np.asarray(var_index)
        self.var_exp = np.maximum(0, self.var_index - 1) # 0 for differential components

        self.current_jac = True
        self.LU = None

        # variables for implementing first-same-as-last
        self.z_scaled = self.f
    
    def _validate_mass_matrix(self, mass_matrix):
        if mass_matrix is None:
            M = self.I
            # index_algebraic_vars = None
            index_algebraic_vars = np.array([], dtype=int)
            nvars_algebraic = 0
        elif callable(mass_matrix):
            raise ValueError("`mass_matrix` should be a constant matrix, but is"
                             " callable")
        else:
            if issparse(mass_matrix):
                M = csc_matrix(mass_matrix)
                index_algebraic_vars = np.where(np.all(M.toarray()==0, axis=1))[0]
            else:
                M = np.asarray(mass_matrix, dtype=float)
                index_algebraic_vars = np.where(np.all(M==0, axis=1))[0]
            if M.shape != (self.n, self.n):
                raise ValueError("`mass_matrix` is expected to have shape {}, "
                                 "but actually has {}."
                                 .format((self.n, self.n), M.shape))
            nvars_algebraic = index_algebraic_vars.size

        return M, index_algebraic_vars, nvars_algebraic

    def _validate_jac(self, jac, sparsity):
        t0 = self.t
        y0 = self.y

        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, f):
                self.njev += 1
                J, self.jac_factor = num_jac(self.fun_vectorized, t, y, f,
                                             self.atol, self.jac_factor, sparsity)

                return J

            J = jac_wrapped(t0, y0, self.f)
        elif callable(jac):
            J = jac(t0, y0)
            self.njev = 1
            if issparse(J):
                J = csc_matrix(J)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return csc_matrix(jac(t, y), dtype=float)
            else:
                J = np.asarray(J, dtype=float)

                def jac_wrapped(t, y, _=None):
                    self.njev += 1
                    return np.asarray(jac(t, y), dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("'jac' is supposed to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
        else:
            if issparse(jac):
                J = csc_matrix(jac)
            else:
                J = np.asarray(jac, dtype=float)

            if J.shape != (self.n, self.n):
                raise ValueError("'jac' is supposed to have shape {}, but "
                                 "actually has {}."
                                 .format((self.n, self.n), J.shape))
            jac_wrapped = None

        return jac_wrapped, J

    def _step_impl(self):
        print(f"t: {self.t}")

        t = self.t
        y = self.y
        f = self.f
        n = y.size

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

        J = self.J
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

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            # TODO: Is there an extrapolation strategy as in Radau?

            # we iterate on the variable z = hf(t, y), as explained in [1]
            # z0 = h_abs * self.z_scaled
            z0 = h_abs * self.fun(t, y) # TODO: This is inefficient!

            # TODO: scaling, see Hairer ???
            scale_newton = atol + rtol * np.abs(z0)
            # # TODO: Is this newton scaling good (yes!)
            # scale_newton /= h**self.var_exp

            converged_tr = False
            # TR stage
            while not converged_tr:
                if LU is None:
                    # LU = self.lu(self.I - d * h_abs * J)
                    LU = self.lu(self.mass_matrix - d * h_abs * J)

                t_gm = t + h * gm
                # TODO: y + d * z0 can be computed only once here and be passed as some y0 to the solve_trbdf2_system function.
                # fun_TR = lambda z: h * self.fun(t_gm, y + d * z0 + d * z) - z
                fun_TR = lambda z: h * self.fun(t_gm, y + d * z0 + d * z) - np.dot(self.mass_matrix, z)
                converged_tr, n_iter_tr, z_tr, rate_tr = solve_trbdf2_system(
                    fun_TR, z0, scale_newton, self.newton_tol, LU, self.solve_lu, self.mass_matrix,
                )

                if not converged_tr:
                    if current_jac:
                        break
                    J = self.jac(t, y, f)
                    current_jac = True
                    LU = None

            if not converged_tr:
                h_abs *= 0.5
                LU = None
                continue

            y_tr = y + d * z0 + d * z_tr

            # BDF stage
            converged_bdf = False
            while not converged_bdf:
                if LU is None:
                    # LU = self.lu(self.I - d * h_abs * J)
                    LU = self.lu(self.mass_matrix - d * h_abs * J)

                # TODO: Find reference for this predictor. Is there a predictor for the TR step as well?
                z_bdf0 = pd0 * z0 + pd1 * z_tr + pd2 * (y_tr - y)
                # fun_bdf = lambda z: h * self.fun(t_new, y + w * z0 + w * z_tr + d * z) - z
                fun_bdf = lambda z: h * self.fun(t_new, y + w * z0 + w * z_tr + d * z) - np.dot(self.mass_matrix, z)

                converged_bdf, n_iter_bdf, z_bdf, rate_bdf = solve_trbdf2_system(
                    fun_bdf, z_bdf0, scale_newton, self.newton_tol, LU, self.solve_lu, self.mass_matrix,
                )

                if not converged_bdf:
                    if current_jac:
                        break
                    J = self.jac(t_gm, y_tr, f)
                    LU = None
                    current_jac = True

            if not converged_bdf:
                h_abs *= 0.5
                LU = None
                continue

            y_new = y + w * z0 + w * z_tr + d * z_bdf

            n_iter = max(n_iter_tr, n_iter_bdf)
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            # scale = atol + rtol * np.abs(y_new)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            # TODO:
            # scale /= h**self.var_exp
            # error = 0.5 * (y + e0 * z0 + e1 * z_tr + e2 * z_bdf - y_new)
            error = ((e0 - w) * z0 + (e1 - w) * z_tr + (e2 - d) * z_bdf)

            # always use stabilized error, see Hosea ???
            # stabilised_error = self.solve_lu(LU, error)
            stabilised_error = self.solve_lu(LU, self.mass_matrix.dot(error))
            # stabilised_error = error
            # # see [1], chapter IV.8, page 127
            # stabilised_error = self.solve_lu(LU_real, f + self.mass_matrix.dot(ZE))
            error_norm = norm(stabilised_error / scale)

            if rejected and error_norm > 1: # try with stabilised error estimate
                print("second stabilized error")
                # stabilised_error = self.solve_lu(LU, stabilised_error)
                stabilised_error = self.solve_lu(LU, self.mass_matrix.dot(stabilised_error))
                # stabilised_error = error
                error_norm = norm(stabilised_error / scale)

            if error_norm > 1:
                factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
                h_abs *= max(MIN_FACTOR, safety * factor)

                LU = None
                rejected = True
            else:
                step_accepted = True

        # Step is converged and accepted
        rate = min(rate_tr, rate_bdf)
        recompute_jac = jac is not None and n_iter > 2 and rate > 1e-3

        factor = predict_factor(h_abs, h_abs_old, error_norm, error_norm_old)
        factor = min(MAX_FACTOR, safety * factor)

        if not recompute_jac and factor < 1.2:
            factor = 1
        else:
            LU = None

        f_new = self.fun(t_new, y_new)
        if recompute_jac:
            J = jac(t_new, y_new, f_new)
            current_jac = True
        elif jac is not None:
            current_jac = False

        self.h_abs_old = self.h_abs
        self.error_norm_old = error_norm

        self.h_abs = h_abs * factor

        self.y_old = y

        self.t = t_new
        self.y = y_new
        self.f = f_new

        self.z_scaled = z_bdf / self.h_abs_old

        self.LU = LU
        self.current_jac = current_jac
        self.J = J

        self.t_old = t
        # self.sol = self._compute_dense_output()

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
    whether t <= t0 + gm*h or t > t0 + gm*h

    .. [1] Hosea, M. E., and L. F. Shampine. "Analysis and implementation of TR-BDF2."
           Applied Numerical Mathematics 20.1-2 (1996): 21-37.
    """

    def __init__(self, t0, t, h, z0, z_tr, z_bdf, y0, y_tr, y_bdf):
        super().__init__(t0, t)
        self.t0 = t0
        self.h = h
        # coefficients for the case t <= t_old + gm*h (denoted as v10, v11...)
        v10 = y0
        v11 = gm * z0
        v12 = y_tr - y0 - v11
        v13 = gm * (z_tr - z0)
        self.V1 = np.array([v10, v11, 3 * v12 - v13, v13 - 2 * v12])

        # coefficients for the case t > t0 + gm*h
        v20 = y_tr
        v21 = (1 - gm) * z_tr
        v22 = y_bdf - y_tr - v21
        v23 = (1 - gm) * (z_bdf - z_tr)
        self.V2 = np.array([v20, v21, 3 * v22 - v23, v23 - 2 * v22])

    def _call_impl(self, t):
        if t.ndim == 0:
            if t <= self.t0 + gm * self.h:
                r = (t - self.t0) / (gm * self.h)
                V = self.V1
            else:
                t_tr = self.t0 + gm * self.h
                V = self.V2
                r = (t - t_tr) / ((1 - gm) * self.h)

            R = np.cumprod([1, r, r, r])
            y = np.dot(V.T, R)
        else:
            y = []
            for tk in t:
                y.append(self._call_impl(tk))
            y = np.array(y).T
        return y

