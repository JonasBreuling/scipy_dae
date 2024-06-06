import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import time
from PyDAE._scipy.integrate._dae.dae import solve_dae, Radau


"""RHS of stiff van der Pol equation, see mathworks.
References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-stiff-odes.html
"""

def f(t, y, mu=1e3):
    y1, y2 = y

    yp = np.zeros(2, dtype=y.dtype)
    yp[0] = y2
    yp[1] = mu * (1 - y1 * y1) * y2 - y1

    return yp

def F(t, y, yp, mu=1e3):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros(2, dtype=y.dtype)
    F[0] = y1p - y2
    F[1] = y2p - mu * (1 - y1 * y1) * y2 - y1

    return F


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e2
    # t1 = 3e3
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([2, 0], dtype=float)
    yp0 = f(t0, y0)

    # solver options
    atol = rtol = 1e-3

    ####################
    # reference solution
    ####################
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    t_scipy = sol.t
    y_scipy = sol.y

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
    # TRBDF2:
    # # - nfev: 3409
    # # - njev: 17
    # # - nlu: 86
    # - nfev: 2442
    # - njev: 41
    # - nlu: 104
    # - nfev: 1987
    # - njev: 45
    # - nlu: 110
    # - nfev: 1674
    # - njev: 26
    # - nlu: 85
    # Radau:
    # - nfev: 1397
    # - njev: 16
    # - nlu: 110
    # BDF:
    # - nfev: 633
    # - njev: 6
    # - nlu: 55

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
