import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae


"""Clairaut's implicit differential equation, see wikipedia.

References:
-----------
wikipedia: https://en.wikipedia.org/wiki/Clairaut%27s_equation
"""
f = lambda x: np.log(x)
fp = lambda x: 1 / x
fpp = lambda x: -1 / x**2
# f = lambda x: x**2
# fp = lambda x: x
# fpp = lambda x: np.ones_like(x)


def F(t, y, yp):
    return (
        y - t * yp - f(yp)
    )


def t(xi):
    return -f(xi)

def true_sol(xi):
    # xi = -1 / t
    # return (
    #     np.atleast_1d(np.log(xi) - 1), 
    #     np.atleast_1d(xi),
    # )
    return (
        np.atleast_1d(f(xi) - xi * fp(xi)), 
        np.atleast_1d(fp(xi) - fp(xi) - -xi * fpp(xi)),
    )


if __name__ == "__main__":
    # time span
    xi0 = 1
    xi1 = 2
    t0 = t(xi0)
    t1 = t(xi1)
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = true_sol(xi0)
    yp0 = np.ones_like(y0)

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
    ax.plot(t, true_sol(t)[0], "-ok", label="y_true")
    ax.plot(t, y[0], "--xr", label=f"y {method}")
    ax.grid()
    ax.legend()

    plt.show()
