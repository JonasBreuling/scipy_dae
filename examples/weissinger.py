import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae


"""Weissinger's implicit differential equation, see mathworks.

References:
-----------
mathworks: https://www.mathworks.com/help/matlab/ref/ode15i.html#bu7u4dt-1
"""
def F(t, y, yp):
    return (
        t * y**2 * yp**3 
        - y**3 * yp**2 
        + t * (t**2 + 1) * yp 
        - t**2 * y
    )

def jac(t, y, yp):
    Jy = np.array([
        2 * t * y * yp**3
        - 3 * y**2 * yp**2  
        - t**2 * y,
    ])
    Jyp = np.array([
        3 * t * y**2 * yp**2 
        - 2 * y**3 * yp 
        + t * (t**2 + 1)
    ])
    return Jy, Jyp


if __name__ == "__main__":
    # time span
    t0 = 1
    t1 = 10
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([np.sqrt(3 / 2)])
    yp0 = np.array([np.sqrt(6) / 3])

    # solver options
    atol = rtol = 1e-6

    # run the solver
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method)
    end = time.time()
    t = sol.t
    y = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"message: {message}")
    print(f"elapsed time: {end - start}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # visualization
    fig, ax = plt.subplots()

    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.plot(t, y[0], "-ok")
    ax.grid()

    plt.show()
