import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate._ivp.common import select_initial_step, norm
from cardillo.math.fsolve import fsolve
from cardillo.solver import SolverOptions


NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

def tableau():
    A = np.eye(1)
    b = np.ones(1)
    c = np.ones(1)
    p = 1
    return A, b, c, p

def tableau():
    A = np.array([
        [5 / 12, -1 / 12],
        [3 / 4, 1 / 4],
    ])
    b = A[-1, :]
    c = np.array([1 / 3, 1])
    p = 3
    return A, b, c, p

def euler(
        fun, y0, t_span, rtol=1e-3, atol=1e-6, mass_matrix=None, var_index=None
):
    ny = y0.shape[0]
    if mass_matrix is None:
        mass_matrix = np.eye(ny)

    t_start, t_finish = t_span
    f0 = fun(t_start, y0)
    h = select_initial_step(fun, t_start, y0, f0, direction=1, order=1, rtol=rtol, atol=atol)

    sol_t = [t_start]
    sol_y = [y0.copy()]

    A, b, c, p = tableau()
    s = len(b)

    def step(t0, y0, h):        
        def R(Y):
            Y = Y.reshape(s, -1, order="C")
            fs = np.array([
                h * fun(t0 + c[i] * h, Y[i]) for i in range(s)
            ])
            return np.concatenate([
                mass_matrix @ (Y[i] - y0) - A[i] @ fs for i in range(s)
            ]).reshape(-1)

        Y0 = np.tile(y0, s)
        sol = fsolve(R, Y0, options=SolverOptions(numerical_jacobian_method="2-point"))
        Y = sol.x
        Y = Y.reshape(s, -1, order="C")
        y = Y[-1]
        t = t0 + h
        nit = sol.nit
        success = sol.success
        return t, y, nit, success
    
    t = t_start
    y = y0.copy()
    while t < t_finish:
        step_accepted = False
        while not step_accepted:
            t1, y1, n_iter1, success1 = step(t, y, h)
            if not success1:
                h *= 0.5
                continue

            t_half, y_half, n_iter_half, success_half = step(t, y, h / 2)
            if not success_half:
                h *= 0.5
                continue

            t2, y2, n_iter2, success2 = step(t_half, y_half, h / 2)
            if not success2:
                h *= 0.5
                continue

            scale = atol + np.maximum(np.abs(y0), np.abs(y1)) * rtol
            # scale /= h**np.maximum(var_index - 1, 0)
            error = (y2 - y1) / (2**p - 1)
            error_norm = norm(error / scale)

            n_iter = np.max([n_iter1, n_iter_half, n_iter2])
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)

            if error_norm > 1:
                with np.errstate(divide='ignore'):
                    factor = error_norm ** (-1 / p)
                h *= max(MIN_FACTOR, safety * factor)
            else:
                step_accepted = True

            with np.errstate(divide='ignore'):
                factor = error_norm ** (-1 / p)
            factor = min(MAX_FACTOR, safety * factor)
            h *= factor

            # # advance with half-step method
            # # TODO: This is no good idea
            t = t2
            y = y2.copy()
            # # use full step to get correct error estmate!
            # t = t1
            # y = y1.copy()
            print(f"t: {t:0.2e}/{t_finish:0.2e}; h: {h:0.2e}")
            
            sol_t.append(t)
            sol_y.append(y.copy())

    return np.array(sol_t), np.array(sol_y)
