import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import time
from PyDAE._scipy.integrate._dae.dae import solve_dae, Radau

k = -1
# k = 1

def f(t, y):
    # return k * y
    # return np.sin(t) * y
    return np.sin(y) + np.cos(t) * y

def F(t, y, yp):
    return yp - f(t, y)


if __name__ == "__main__":
    # time span
    t0 = 0
    # t1 = 20
    t1 = 2
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([-0.5, 0.5], dtype=float)
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
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, first_step=1e-1)
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

    ax[0].plot(t, y[0], "-ok", label=f"y1 ({method.__name__})", mfc="none")
    ax[0].plot(t_scipy, y_scipy[0], "-xr", label="y1 scipy")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "-ok", label=f"y2 ({method.__name__})", mfc="none")
    ax[1].plot(t_scipy, y_scipy[1], "-xr", label="y2 scipy")
    ax[1].legend()
    ax[1].grid()

    plt.show()
