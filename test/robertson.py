import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import time
from PyDAE._scipy.integrate._dae.dae import solve_dae, Radau


"""Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011
"""

def f(t, y):
    y1, y2, y3 = y

    yp = np.zeros(3, dtype=y.dtype)
    yp[0] = -0.04 * y1 + 1e4 * y2 * y3
    yp[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    yp[2] = 3e7 * y2**2

    return yp

def F(t, y, yp):
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=y.dtype)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e1
    # t1 = 1e3
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    yp0 = f(t0, y0)

    # solver options
    atol = rtol = 1e-5

    ####################
    # reference solution
    ####################
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    t_scipy = sol.t
    y_scipy = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    ##############
    # dae solution
    ##############
    # method = BDF
    method = Radau
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "-ok", label="y1 DAE" + f" ({method.__name__})", mfc="none")
    ax.plot(t, y[1] * 1e4, "-ob", label="y2 DAE" + f" ({method.__name__})", mfc="none")
    ax.plot(t, y[2], "-og", label="y3 DAE" + f" ({method.__name__})", mfc="none")
    ax.plot(t_scipy, y_scipy[0], "xr", label="y1 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[1] * 1e4, "xy", label="y2 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[2], "xm", label="y3 scipy Radau", markersize=7)
    ax.set_xscale("log")
    ax.legend()
    ax.grid()

    plt.show()
