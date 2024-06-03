import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import time
from dae import BDF, Radau, TRBDF2

def rhs(t, y, mu=1e3):
    """RHS of stiff van der Pol equation, see mathworks.
    References:
    -----------
    mathworks: https://de.mathworks.com/help/matlab/math/solve-stiff-odes.html
    """
    y1, y2 = y

    y_dot = np.zeros(2, dtype=y.dtype)
    y_dot[0] = y2
    y_dot[1] = mu * (1 - y1 * y1) * y2 - y1

    return y_dot

if __name__ == "__main__":

    # time span
    t0 = 0
    # t1 = 3e3
    t1 = 1e3
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([2, 0], dtype=float)

    # solver options
    rtol = atol = 1e-6
    # rtol = atol = 1e-8
    # rtol = atol = 1e-10

    # reference solution
    sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    t_scipy = sol.t
    y_scipy = sol.y

    # dae solution
    method = BDF
    # method = Radau
    # method = TRBDF2
    start = time.time()
    sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method=method, var_index=[0, 0])
    end = time.time()
    print(f"elapsed time: {end - start}")
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")
    t = sol.t
    y = sol.y

    # from dae.euler import euler
    # t, y = euler(rhs, y0, t_span, rtol, atol)
    # y = y.T

    # visualization
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[0], "-ok", label=f"y ({method.__name__})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "-xr", label="y scipy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "-ok", label=f"y_dot ({method.__name__})", mfc="none")
    ax[1].plot(t_scipy, y_scipy[1], "-xr", label="y_dot scipy")
    ax[1].legend()
    ax[1].grid()

    plt.show()
