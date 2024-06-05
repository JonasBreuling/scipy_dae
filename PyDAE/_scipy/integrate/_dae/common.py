import numpy as np


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
