import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""Index 2 DAE found in Section 3.1 of Brenan1996.

References:
-----------
Brenan1996: https://doi.org/10.1137/1.9781611971224.ch4
"""

eta = -2

def g(t):
    return np.sin(t)

def gp(t):
    return np.cos(t)

def gpp(t):
    return -np.sin(t)


# TODO: How to perform index reduction?
def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    # index 2
    # F[0] = y1 + eta * t * y2 - g(t)
    # index 1
    F[0] = y1p + eta * y2 + eta * t * y2p - gp(t)
    F[1] = y1p + eta * t * y2p + (1 + eta) * y2

    return F


def true_sol(t):
    return (
        np.array(
            [
                g(t) + eta * t * gp(t),
                -gp(t),
            ]
        ),
        np.array(
            [
                gp(t) + eta * gp(t) + eta * t * gpp(t),
                -gpp(t),
            ]
        ),
    )


if __name__ == "__main__":
    # time span
    t0 = 1
    t1 = 1e1
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-6

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)
    y0, yp0 = true_sol(t0)

    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {np.linalg.norm(F(t0, y0, yp0))}")
    # y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    start = time.time()
    method = "BDF"
    # method = "Radau"
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    y = sol.y
    tp = t
    yp = sol.yp
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")
    print(f"y[:, -1]: {y[:, -1]}")

    # error
    diff = y[:, -1] - np.array([
        np.exp(-t1) + t1 * np.sin(t1),
        np.sin(t1),
    ])
    error = np.linalg.norm(diff)
    print(f"error: {error}")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--or", label="y1")
    ax.plot(t, y[1], "--og", label="y2")

    ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    ax.plot(t, np.sin(t), "-g", label="y2 true")

    ax.grid()
    ax.legend()

    plt.show()
