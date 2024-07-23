import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""Index 1 DAE found in Chapter 4 of Brenan1996.

References:
-----------
Brenan1996: https://doi.org/10.1137/1.9781611971224.ch
"""
def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros_like(y, dtype=np.common_type(y, yp))
    F[0] = y1p - t * y2p + y1 - (1 + t) * y2
    F[1] = y2 - np.sin(t)

    return F


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 5
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-6

    # initial conditions
    y0 = np.array([1, 0], dtype=float)
    yp0 = np.array([-1, 1], dtype=float)

    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    start = time.time()
    # method = "BDF"
    method = "Radau"
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

    # errors
    dt = t[1] - t[0]
    error_y1 = np.linalg.norm((y[0] - np.exp(-t) + t * np.sin(t)) * dt)
    error_y2 = np.linalg.norm((y[1] - np.sin(t)) * dt)
    print(f"error: [{error_y1}, {error_y2}]")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, y[1], "--g", label="y2")

    ax.plot(t, np.exp(-t) + t * np.sin(t), "-r", label="y1 true")
    ax.plot(t, np.sin(t), "-g", label="y2 true")

    ax.grid()
    ax.legend()

    plt.show()
