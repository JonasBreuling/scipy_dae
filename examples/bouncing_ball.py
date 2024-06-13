import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae
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
        # prox_la = fb(lap, u)
        prox_la = min(lap, u)
        # prox_la = u
        print(f"u: {u}")
    else:
        prox_la = lap
    # prox_mu = fb(mup, y)
    prox_mu = min(mup, y)
    prox_mu = mup
    
    return np.array([
                yp - (u + mup),
                mass * up - (-mass * gravity + lap),
                prox_mu,
                prox_la,
        ])

def jac(t, y, yp, f=None):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                            z, method="3-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 3
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-1

    # initial conditions
    y0 = np.array([1, 0, 0, 0], dtype=float)
    yp0 = np.array([0, -gravity, 0, 0], dtype=float)

    # F0 = F(t0, y0, yp0)
    # print(f"F0: {F0}")
    # exit()

    ##############
    # dae solution
    ##############
    # method = "BDF"
    method = "Radau"
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, max_step=1e-1, jac=jac)
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
