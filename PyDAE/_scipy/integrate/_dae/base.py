import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput


def check_arguments(fun, y0, yp0, support_complex):
    """Helper function for checking arguments common to all solvers."""
    y0 = np.asarray(y0)
    yp0 = np.asarray(yp0)
    if np.issubdtype(np.common_type(y0, yp0), np.complexfloating):
    # if np.issubdtype(y0.dtype, np.complexfloating) or np.issubdtype(yp0.dtype, np.complexfloating):
        if not support_complex:
            raise ValueError("`y0` or `yp0` is complex, but the chosen "
                             "solver does not support integration in a "
                             "complex domain.")
        dtype = complex
    else:
        dtype = float
    y0 = y0.astype(dtype, copy=False)
    yp0 = yp0.astype(dtype, copy=False)

    if y0.ndim != 1:
        raise ValueError("`y0` must be 1-dimensional.")
    if yp0.ndim != 1:
        raise ValueError("`yp0` must be 1-dimensional.")
    
    if y0.shape != yp0.shape:
        raise ValueError("`y0` and `yp0` must be of same shape.")

    if not np.isfinite(y0).all():
        raise ValueError("All components of the initial state `y0` must be finite.")
    if not np.isfinite(yp0).all():
        raise ValueError("All components of the initial state `yp0` must be finite.")

    def fun_wrapped(t, y, yp):
        return np.asarray(fun(t, y, yp), dtype=dtype)

    return fun_wrapped, y0, yp0

# TODO: Add check_jacobi here and build numerical Jacobian otherwise


# TODO: Add jac_y and jac_yp here, otherwise we cannot check these in the base class.
class DaeSolver:
    """Base class for DAE solvers.

    In order to implement a new solver you need to follow the guidelines:

        1. A constructor must accept parameters presented in the base class
           (listed below) along with any other parameters specific to a solver.
        2. A constructor must accept arbitrary extraneous arguments
           ``**extraneous``, but warn that these arguments are irrelevant
           using `common.warn_extraneous` function. Do not pass these
           arguments to the base class.
        3. A solver must implement a private method `_step_impl(self)` which
           propagates a solver one step further. It must return tuple
           ``(success, message)``, where ``success`` is a boolean indicating
           whether a step was successful, and ``message`` is a string
           containing description of a failure if a step failed or None
           otherwise.
        4. A solver must implement a private method `_dense_output_impl(self)`,
           which returns a `DenseOutput` object covering the last successful
           step.
        5. A solver must have attributes listed below in Attributes section.
           Note that ``t_old`` and ``step_size`` are updated automatically.
        6. Use `fun(self, t, y, yp)` method for the system evaluation, this
           way the number of function evaluations (`nfev`) will be tracked
           automatically.
        7. For convenience, a base class provides `fun_single(self, t, y, yp)`
           and `fun_vectorized(self, t, y, yp)` for evaluating the system in
           non-vectorized and vectorized fashions respectively (regardless of
           how `fun` from the constructor is implemented). These calls don't
           increment `nfev`.
        8. If a solver uses a Jacobian matrix and LU decompositions, it should
           track the number of Jacobian evaluations (`njev`) and the number of
           LU decompositions (`nlu`).
        9. By convention, the function evaluations used to compute a finite
           difference approximation of the Jacobian should not be counted in
           `nfev`, thus use `fun_single(self, t, y, yp)` or
           `fun_vectorized(self, t, y, yp)` when computing a finite 
           difference approximation of the Jacobian.

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
    yp0 : array_like, shape (n,), optional
        Initial derivative. If the derivative is not given by the user, it is 
        esimated using??? => TODO: See Petzold/ Shampine and coworkers how 
        this is done.
    t_bound : float
        Boundary time --- the integration won't continue beyond it. It also
        determines the direction of the integration.
    var_index : array_like, shape (n,), optional
        Differentiation index of the respective row of f(t, y, y') = 0. 
        Depending on this index, the error estimates are scaled by the 
        stepsize h**(index - 1) in order to ensure convergence.
        Default is None, which means all equations are differential equations.
    vectorized : bool
        Whether `fun` can be called in a vectorized fashion. Default is False.

        If ``vectorized`` is False, `fun` will always be called with ``y`` and
        ``yp`` of shape ``(n,)``, where ``n = len(y0) = len(yp0)``.

        If ``vectorized`` is True, `fun` may be called with ``y`` and 
        ``yp`` of shape ``(n, k)``, where ``k`` is an integer. In this 
        case, `fun` must behave such that 
        ``fun(t, y, yp)[:, i] == fun(t, y[:, i], yp[:, i])`` (i.e. each 
        column of the returned array is the defect of the nonlinear equation 
        corresponding with a column of ``y`` and ``yp``.

        Setting ``vectorized=True`` allows for faster finite difference
        approximation of the Jacobian by methods 'Radau' and 'BDF', but
        will result in slower execution for other methods. It can also
        result in slower overall execution for 'Radau' and 'BDF' in some
        circumstances (e.g. small ``len(y0)``).
    support_complex : bool, optional
        Whether integration in a complex domain should be supported.
        Generally determined by a derived solver class capabilities.
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
        Number of the system's rhs evaluations.
    njev : int
        Number of the Jacobian evaluations.
    nlu : int
        Number of LU decompositions.
    """
    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    def __init__(self, fun, t0, y0, yp0, t_bound, vectorized, 
                 support_complex=False):
        self.t_old = None
        self.t = t0
        self._fun, self.y, self.yp = check_arguments(fun, y0, yp0, support_complex)
        self.t_bound = t_bound
        self.vectorized = vectorized

        if vectorized:
            def fun_single(t, y, yp):
                return self._fun(t, y[:, None], yp[:, None]).ravel()
            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            def fun_vectorized(t, y, yp):
                f = np.empty_like(y)
                for i, (yi, ypi) in enumerate(zip(y.T, yp.T)):
                    f[:, i] = self._fun(t, yi, ypi)
                return f

        def fun(t, y, yp):
            self.nfev += 1
            return self.fun_single(t, y, yp)

        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        self.status = 'running'

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    def _validate_jac(self, jac, sparsity, wrt_y=True):
        # TODO: I'm not sure if this can be done in the base class since 
        # depending on the method y depends on yp or vice versa.
        t0 = self.t
        y0 = self.y
        yp0 = self.yp
        if jac is None:
            if sparsity is not None:
                if issparse(sparsity):
                    sparsity = csc_matrix(sparsity)
                groups = group_columns(sparsity)
                sparsity = (sparsity, groups)

            def jac_wrapped(t, y, yp):
                self.njev += 1
                f = self.fun_single(t, y, yp)
                # TODO: Maybe check both derivatives at the same time. Hence, 
                # this will be demistfied.
                if wrt_y:
                    J, self.jac_factor = num_jac(
                        lambda t, y: self.fun_vectorized(t, y, yp), t, y, f,
                        self.atol, self.jac_factor, sparsity,
                    )
                else:
                    J, self.jac_factor = num_jac(
                        lambda t, yp: self.fun_vectorized(t, y, yp), t, yp, f,
                        self.atol, self.jac_factor, sparsity,
                    )
                return J
            J = jac_wrapped(t0, y0, yp0)
        elif callable(jac):
            J = jac(t0, y0, yp0)
            self.njev += 1
            if issparse(J):
                J = csc_matrix(J, dtype=np.common_type(y0, yp0))

                def jac_wrapped(t, y, yp):
                    self.njev += 1
                    return csc_matrix(jac(t, y, yp), dtype=np.common_type(y0, yp0))
            else:
                J = np.asarray(J, dtype=np.common_type(y0, yp0))

                def jac_wrapped(t, y, yp):
                    self.njev += 1
                    return np.asarray(jac(t, y, yp), dtype=np.common_type(y0, yp0))

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                    "actually has {}."
                                    .format((self.n, self.n), J.shape))
        else:
            if issparse(jac):
                J = csc_matrix(jac, dtype=np.common_type(y0, yp0))
            else:
                J = np.asarray(jac, dtype=np.common_type(y0, yp0))

            if J.shape != (self.n, self.n):
                raise ValueError("`jac` is expected to have shape {}, but "
                                    "actually has {}."
                                    .format((self.n, self.n), J.shape))
            jac_wrapped = None

        return jac_wrapped, J

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != 'running':
            raise RuntimeError("Attempt to step on a failed or finished "
                               "solver.")

        if self.n == 0 or self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = 'finished'
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = 'failed'
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = 'finished'

        return message

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        if self.t_old is None:
            raise RuntimeError("Dense output is available after a successful "
                               "step was made.")

        if self.n == 0 or self.t == self.t_old:
            # Handle corner cases of empty solver and no integration.
            return ConstantDenseOutput(self.t_old, self.t, self.y)
        else:
            return self._dense_output_impl()

    def _step_impl(self):
        raise NotImplementedError

    def _dense_output_impl(self):
        raise NotImplementedError
