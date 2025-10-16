import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


"""Modified index 2 DAE found in Jay1993 Example 7.

References:
-----------
Jay1993: https://link.springer.com/article/10.1007/BF01990349
"""
omega = 2 * np.pi
def F(t, y, yp):
    y1, la = y
    y1p, lap = yp
    la = lap # index reduction

    F = np.zeros(2, dtype=y.dtype)
    F[0] = y1p - la
    F[1] = y1 - np.sin(omega * t)

    return F

def sol_true(t):
    y = np.array([
        np.sin(omega * t), 
        omega * np.cos(omega * t),
    ])
    yp = np.array([
        omega * np.cos(omega * t),
        -omega**2 * np.sin(omega * t),
    ])
    return y, yp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 2
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))

    # tolerances
    rtol = atol = 1e-12

    # initial conditions
    y0, yp0 = sol_true(t0)

    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")

    ##############
    # dae solution
    ##############
    start = time.time()
    # method = "BDF"
    method = "Radau"
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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

    # exit()

    # # errors
    # dt = t[1] - t[0]
    # error_y1 = np.linalg.norm((y[0] - np.exp(t)) * dt)
    # error_y2 = np.linalg.norm((y[1] - np.exp(-2 * t)) * dt)
    # # Note: We have an expected order reduction of h here since we control the
    # #       error of y[2] instead of yp[2].
    # # error_z = np.linalg.norm((yp[2] - np.exp(2 * tp)) * dt)
    # error_z = np.linalg.norm((yp[2] - np.exp(2 * tp)) * dt**2)
    # print(f"error: [{error_y1}, {error_y2}, {error_z}]")

    # visualization
    fig, ax = plt.subplots()

    y_true, yp_true = sol_true(t)

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, yp[1], "--g", label="la")

    ax.plot(t, y_true[0], "-r", label="y1 true")
    ax.plot(t, y_true[1], "-g", label="la true")

    ax.grid()
    ax.legend()

    plt.show()
