######################################
# derived from
# from scipy.integrate._ivp import ivp
######################################

import inspect
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult, prepare_events, handle_events, find_active_events
from scipy.integrate._ivp.common import OdeSolution
from .base import DaeSolver
from .bdf import BDFDAE
from .radau import RadauDAE


METHODS = {
    "BDF": BDFDAE,
    "Radau": RadauDAE,
}


MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}


# TODO:
# - expect consistent initial conditions and add a helper function that 
#   computes them as done by matlab?
# - add events depending on y'(t)?
# - dense output for y'
def solve_dae(fun, t_span, y0, y_dot0, method="Radau", t_eval=None, 
              dense_output=False, events=None, vectorized=False, 
              args=None, **options):
    """Solve an initial value problem for a system of differential algebraic 
    equations (DAE's).

    This function numerically integrates a system of implicit ordinary 
    differential equations given an initial value::

        f(t, y, y') = 0
        y(t0) = y0
        y'(t0) = y_dot0

    Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued 
    function (state), y'(t) is an N-D vector-valued function (state 
    derivative) and an N-D vector-valued function f(t, y, y') determines the 
    implicit differential algebraic equations.
    The goal is to find y(t) and y'(t) approximately satisfying the differential
    algebraic equations, given initial values y(t0)=y0 and y'(t0)=y_dot0.

    Some of the solvers support integration in the complex domain, but note
    that the function f must be complex-differentiable (satisfy Cauchy-Riemann 
    equations [11]_).
    To solve a problem in the complex domain, pass y0 or y_dot0 with a complex 
    data type. Another option always available is to rewrite your problem for 
    real and imaginary parts separately.

    Parameters
    ----------
    fun : callable
        Function defining the DAE system: ``f(t, y, y_dot) = 0``. The calling 
        signature is ``fun(t, y, y_dot)``, where ``t`` is a scalar and 
        ``y, y_dot`` are ndarrays with 
        ``len(y) = len(y_dot) = len(y0) = len(y_dot0)``. Additional 
        arguments need to be passed if ``args`` is used (see documentation of
        ``args`` argument). ``fun`` must return an array of the same shape as
        ``y`` and ``y_dot``. See `vectorized` for more information.
    t_span : 2-member sequence
        Interval of integration (t0, t_bound). The solver starts with t=t0 and
        integrates until it reaches t=t_bound. Both t0 and t_bound must be 
        floats or values interpretable by the float conversion function.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    y_dot0 : array_like, shape (n,)
        Initial derivative. For problems in the complex domain, pass `y_dot0` 
        with a complex data type (even if the initial value is purely real).
    method : string or `DaeSolver`, optional
        Integration method to use:

            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
              order 5 [4]_. The error is controlled with a third-order accurate
              embedded formula. A cubic polynomial which satisfies the
              collocation conditions is used for the dense output.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the
              complex domain.

        You can also pass an arbitrary class derived from `DaeSolver` which
        implements the solver.
    t_eval : array_like or None, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool, optional
        Whether to compute a continuous solution. Default is False.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` where
        additional argument have to be passed if ``args`` is used (see
        documentation of ``args`` argument). Each function must return a
        float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default,
        all zeros will be found. The solver looks for a sign change over
        each step, so if multiple zero crossings occur within one step,
        events may be missed. Additionally each `event` function might
        have the following attributes:

            terminal: bool or int, optional
                When boolean, whether to terminate integration if this event occurs.
                When integral, termination occurs after the specified the number of
                occurences of this event.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    # vectorized : bool, optional
    #     Whether `fun` can be called in a vectorized fashion. Default is False.

    #     If ``vectorized`` is False, `fun` will always be called with ``y`` of
    #     shape ``(n,)``, where ``n = len(y0)``.

    #     If ``vectorized`` is True, `fun` may be called with ``y`` of shape
    #     ``(n, k)``, where ``k`` is an integer. In this case, `fun` must behave
    #     such that ``fun(t, y)[:, i] == fun(t, y[:, i])`` (i.e. each column of
    #     the returned array is the time derivative of the state corresponding
    #     with a column of ``y``).

    #     Setting ``vectorized=True`` allows for faster finite difference
    #     approximation of the Jacobian by methods 'Radau' and 'BDF', but
    #     will result in slower execution for other methods and for 'Radau' and
    #     'BDF' in some circumstances (e.g. small ``len(y0)``).
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    **options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
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
    jac : (array_like, array_like), (sparse_matrix, sparse_matrix), callable or None, optional
        Jacobian matrices of the right-hand side of the system with respect
        to y and y', required by the 'Radau' and 'BDF' method. The
        Jacobian matrices have shape (n, n) and their elements (i, j) are equal to
        ``d f_i / d y_j`` and ``d f_i / d y_j'``, respectively.  There are 
        three ways to define the Jacobian:

            * If (array_like, array_like) or (sparse_matrix, sparse_matrix) 
              the Jacobian matrices are assumed to be constant.
            # TODO: Add constant J_y'!
            * If callable, the Jacobians are assumed to depend on both
              t, y and y'; it will be called as ``jac(t, y, y')``, as necessary.
              Additional arguments have to be passed if ``args`` is
              used (see documentation of ``args`` argument).
              For 'Radau' and 'BDF' methods, the return value might be a
              tuple of sparse matrices.
            * If None (default), the Jacobians will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobians rather than
        relying on a finite-difference approximation.
    jac_sparsity : array_like, sparse matrix or None, optional
        Defines a sparsity structure of the Jacobian matrix for a finite-
        difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few
        non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations [10]_. A zero entry means that
        a corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    # TODO: Add number of solve LGS's as done by matlab.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.

    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).

    References
    ----------
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Backward Differentiation Formula
            <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
            on Wikipedia.
    .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
           COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
    .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
           Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
           pp. 55-64, 1983.
    .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
           nonstiff systems of ordinary differential equations", SIAM Journal
           on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
           1983.
    .. [9] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    .. [10] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
            sparse Jacobian matrices", Journal of the Institute of Mathematics
            and its Applications, 13, pp. 117-120, 1974.
    .. [11] `Cauchy-Riemann equations
             <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
             Wikipedia.
    .. [13] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
            Equations I: Nonstiff Problems", Sec. II.
    .. [14] `Page with original Fortran code of Radau
            <http://www.unige.ch/~hairer/software.html>`_.


    """
    if method not in METHODS and not (
            inspect.isclass(method) and issubclass(method, DaeSolver)):
        raise ValueError(f"`method` must be one of {METHODS} or DaeSolver class.")

    t0, t_bound = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters. Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        def fun(t, x, x_dot, fun=fun):
            return fun(t, x, x_dot, *args)
        
        # TODO: Add validate jac here already, since Jacobians are required for all methods!
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x, x_dot: jac(t, x, x_dot, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, t_bound)) or np.any(t_eval > max(t0, t_bound)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if t_bound > t0 and np.any(d <= 0) or t_bound < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if t_bound > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    solver = method(fun, t0, y0, y_dot0, t_bound, vectorized=vectorized, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
        yps = [y_dot0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
        yps = []
    else:
        ts = []
        ys = []
        yps = []

    interpolants = []

    if events is not None:
        events, max_events, event_dir = prepare_events(events)
        event_count = np.zeros(len(events))
        if args is not None:
            # Wrap user functions in lambdas to hide the additional parameters.
            # The original event function is passed as a keyword argument to the
            # lambda to keep the original function in scope (i.e., avoid the
            # late binding closure "gotcha").
            events = [lambda t, x, event=event: event(t, x, *args)
                      for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y
        yp = solver.yp

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None

        if events is not None:
            raise NotImplementedError("Events are not ready yet")
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                event_count[active_events] += 1
                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, event_count, max_events,
                    t_old, t)

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        if t_eval is None:
            ts.append(t)
            ys.append(y)
            yps.append(yp)
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)

    if t_events is not None:
        # raise NotImplementedError("Events are not ready yet")
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
        yps = np.vstack(yps).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)
        # yps = np.hstack(yps)
        
        # estimate yps via finite differences
        t_final1 = 2 * ts[-1] - ts[-2]
        y_final1 = 2 * ys[:, -1] - ys[:, -2]
        ts1 = np.array([*ts, t_final1])
        ys1 = np.concatenate((ys, y_final1[:, None]), axis=1)
        yps = np.diff(ys1) / np.diff(ts1)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(
                ts, interpolants, #alt_segment=False
                # ts, interpolants, alt_segment=True if method in [BDFDAE] else False
            )
        else:
            sol = OdeSolution(
                ti, interpolants, #alt_segment=False
                # ti, interpolants, alt_segment=True if method in [BDFDAE] else False
            )
    else:
        sol = None

    return OdeResult(t=ts, y=ys, yp=yps, sol=sol, t_events=t_events, y_events=y_events,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status>=0)
