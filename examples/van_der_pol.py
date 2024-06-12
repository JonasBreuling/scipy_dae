import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pydaes.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative


"""RHS of stiff van der Pol equation, see mathworks.
References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-stiff-odes.html
"""

mu = 1e3

def rhs(t, y):
    y1, y2 = y

    yp = np.zeros(2, dtype=y.dtype)
    yp[0] = y2
    yp[1] = mu * (1 - y1 * y1) * y2 - y1

    return yp

def F(t, y, yp):
    return yp - rhs(t, y)


def f(t, z):
    y, yp = z[:2], z[2:]
    return np.concatenate((yp, F(t, y, yp)))

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
    t1 = 3e3
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([2, 0], dtype=float)
    yp0 = rhs(t0, y0)
    z0 = np.concatenate((y0, yp0))

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, jac, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-4

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t_scipy = sol.t
    y_scipy = sol.y
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

    ##############
    # dae solution
    ##############
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
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, y[0], "-ok", label=f"y ({method})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "-xr", label="y scipy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "-ok", label=f"y_dot ({method})", mfc="none")
    ax[1].plot(t_scipy, y_scipy[1], "-xr", label="y_dot scipy")
    ax[1].legend()
    ax[1].grid()

    plt.show()
