import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative


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
    # return yp - f(t, y)
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=y.dtype)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F

def jac(t, y, yp, f):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                            z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e7
    t_span = (t0, t1)
    t_eval = np.logspace(-6, 7, num=1000)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    yp0 = f(t0, y0)

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, jac, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-6

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
    end = time.time()
    print(f"elapsed time: {end - start}")
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
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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

    ax.set_xlabel("t")
    ax.plot(t, y[0], "-ok", label="y1 DAE" + f" ({method})", mfc="none")
    ax.plot(t, y[1] * 1e4, "-ob", label="y2 DAE" + f" ({method})", mfc="none")
    ax.plot(t, y[2], "-og", label="y3 DAE" + f" ({method})", mfc="none")
    ax.plot(t_scipy, y_scipy[0], "xr", label="y1 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[1] * 1e4, "xy", label="y2 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[2], "xm", label="y3 scipy Radau", markersize=7)
    ax.set_xscale("log")
    ax.legend()
    ax.grid()

    plt.show()
