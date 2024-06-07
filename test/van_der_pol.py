import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import time
from PyDAE._scipy.integrate._dae.dae import solve_dae, RadauDAE, BDFDAE #, ODE15I
from PyDAE._scipy.integrate._ivp.radau import Radau
from PyDAE._scipy.integrate._ivp.bdf import BDF


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


mass_matrix = np.diag([1, 1, 0, 0])
# print(f"mass_matrix:\n{mass_matrix}")
# exit()


if __name__ == "__main__":
    # time span
    t0 = 0
    # t1 = 1e1
    # t1 = 1e3
    t1 = 3e3
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([2, 0], dtype=float)
    yp0 = rhs(t0, y0)
    z0 = np.concatenate((y0, yp0))

    # solver options
    atol = rtol = 1e-3

    ####################
    # reference solution
    ####################
    start = time.time()
    # sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method="BDF")
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
    # method = RadauDAE
    method = BDFDAE
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    # # method = Radau
    # method = BDF
    # start = time.time()
    # sol = solve_ivp(f, t_span, z0, atol=atol, rtol=rtol, method=method, mass_matrix=mass_matrix)
    # end = time.time()
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

    ax[0].plot(t, y[0], "-ok", label=f"y ({method.__name__})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "-xr", label="y scipy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "-ok", label=f"y_dot ({method.__name__})", mfc="none")
    ax[1].plot(t_scipy, y_scipy[1], "-xr", label="y_dot scipy")
    ax[1].legend()
    ax[1].grid()

    plt.show()
