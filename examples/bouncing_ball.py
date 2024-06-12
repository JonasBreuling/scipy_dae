import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from scipy_dae.integrate import solve_dae, consistent_initial_conditions, BDFDAE
from scipy.optimize._numdiff import approx_derivative

mass = 1
gravity = 10


def F(t, vy, vyp):
    y, u, la, mu = vy
    yp, up, lap, mup = vyp

    # prox_N = min(lap, y)
    def fb(a, b):
        return a + b - np.sqrt(a**2 + b**2)
    
    # prox_la = mup * fb(lap, u)
    if y <= 0:
        prox_la = fb(lap, u)
    else:
        prox_la = lap
    prox_mu = fb(mup, y)
    
    return np.array([
                yp - (u + mup),
                mass * up - (-mass * gravity + lap),
                prox_mu,
                prox_la,
        ])


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 3
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e2

    # initial conditions
    y0 = np.array([1, 0, 0, 0], dtype=float)
    yp0 = np.array([0, -gravity, 0, 0], dtype=float)

    # F0 = F(t0, y0, yp0)
    # print(f"F0: {F0}")
    # exit()

    ##############
    # dae solution
    ##############
    # method = RadauDAE
    method = BDFDAE
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, max_step=1e-2)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    tp = t[1:]
    yp = np.diff(y) / np.diff(t)
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
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, y[0], "-ok", label="y")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[1], "-ok", label="u")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(tp, yp[2], "-ok", label="la")
    ax[2].legend()
    ax[2].grid()

    plt.show()
