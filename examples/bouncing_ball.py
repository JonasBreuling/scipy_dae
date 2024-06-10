import numpy as np
import time
import matplotlib.pyplot as plt
from PyDAE.integrate._dae.dae import solve_dae, RadauDAE, BDFDAE
from PyDAE.integrate._dae.common import consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative

mass = 1
gravity = 10


def F(t, vy, vyp):
    y, u, la, mu = vy
    yp, up, lap, mup = vyp

    # prox_N = min(lap, y)
    # prox_N = lap + y - np.sqrt(lap**2 + y**2)
    def fb(a, b):
        return a + b - np.sqrt(a**2 + b**2)
    
    # prox_la = mu * fb(lap, u)
    if mup > 0 or y <= 0:
    # if y <= 0:
        prox_la = fb(lap, u)
    else:
        prox_la = lap
    prox_mu = fb(mup, y)
    # prox_mu = fb(mup + la, y)
    # prox_mu = fb(mu + lap, y)

    # eps = 1e-10
    # if y <= 0:
    #     prox_N = fb(lap, u + eps * up)
    # else:
    #     prox_N = lap

    # r = 1e-1
    # prox_arg_N = r * y - lap
    # if prox_arg_N <= 0:
    #     prox_N = y
    #     xi_N = u + eps * u0
    #     prox_arg_N_dot = r * xi_N - la
    #     if prox_arg_N_dot <= 0:
    #         prox_N_dot = xi_N
    #     else:
    #         prox_N_dot = la
    # else:
    #     prox_N = la
    #     prox_N_dot = la
    # # prox_N = min(la, y)
    # # prox_N_dot = (y0 <= 0) * min(la, u + eps * u0)

    # # if y <= 0:
    # #     prox_N_dot = min(la, u + eps * u0) * min(la, y)
    # # else:
    # #     prox_N_dot = min(la, y)
    # # prox_N_dot = min(min(la, y), (y0 <= 0) * min(la, u + eps * u0))
    #     # prox_N_dot = min(la, y)
    # # prox_N_dot = min(la, y) * min(la, u + eps * u0)
    # # prox_N_dot = (mu > 0) * min(la, u + eps * u0)
    # # prox_N_dot = (mu > 0) * min(la, u + eps * u0)

    # # g = np.array([prox_N_dot])
    # g = np.array(
    #     [
    #         prox_N,
    #         prox_N_dot,
    #     ]
    # )

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
