import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate._ivp.tests.test_ivp import compute_error


"""Particle on a circular track subject to tangential force, see Arevalo1995.
   We implement a stabilized index 1 formulation as proposed by Anantharaman1991.

References:
-----------
Arevalo1995: https://link.springer.com/article/10.1007/BF01732606 \\
Anantharaman1991: https://doi.org/10.1002/nme.1620320803
"""
def F(t, vy, vyp):
    # stabilized index 1
    x, y, u, v, _, _ = vy
    x_dot, y_dot, u_dot, v_dot, mup, lap = vyp

    R = np.zeros(6, dtype=np.common_type(vy, vyp))
    R[0] = x_dot - (u + x * mup)
    R[1] = y_dot - (v + y * mup)
    # R[2] = u_dot - (-2 * y + x * lap) # TODO: Compute analytical solution of this example!
    R[2] = u_dot - (2 * y + x * lap)
    R[3] = v_dot - (-2 * x + y * lap)
    R[4] = x * u + y * v
    R[5] = x * x + y * y - 1

    return R

def jac(t, y, yp, f=None):
    n = len(y)
    z = np.concatenate((y, yp))

    def fun_composite(t, z):
        y, yp = z[:n], z[n:]
        return F(t, y, yp)
    
    J = approx_derivative(lambda z: fun_composite(t, z), 
                            z, method="2-point", f0=f)
    J = J.reshape((n, 2 * n))
    Jy, Jyp = J[:, :n], J[:, n:]
    return Jy, Jyp


def sol_true(t):
    y =  np.array([
        np.sin(t**2),
        np.cos(t**2),
        2 * t * np.cos(t**2),
        -2 * t * np.sin(t**2),
        0 * t,
        -4 / 3 * t**3,
    ])

    yp =  np.array([
        2 * t * np.cos(t**2),
        -2 * t * np.sin(t**2),
        -4 * t**2 * np.sin(t**2),
        -4 * t**2 * np.cos(t**2),
        0 * t,
        -4 * t**2,
    ])

    return y, yp


if __name__ == "__main__":
    # time span
    t0 = 1
    t1 = t0 + 10
    t_span = (t0, t1)

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0, yp0 = sol_true(t0)

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, jac, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    atol = rtol = 1e-5

    ##############
    # dae solution
    ##############
    start = time.time()
    jac = None
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, jac=jac)
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

    y_true, _ = sol_true(t)
    _, yp_true = sol_true(tp)
    e = compute_error(y, y_true, rtol, atol)
    ep = compute_error(yp, yp_true, rtol, atol)
    # assert np.all(e < 5)
    # print(f"e: {e}")
    print(f"max(e): {max(e)}")
    # print(f"max(ep): {max(ep)}")
    # exit()

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y[0], "-ok", label="x")
    ax[0].plot(t, y[1], "-ob", label="y")
    ax[0].plot(t, y_true[0], "--xr", label="x_true")
    ax[0].plot(t, y_true[1], "--xg", label="y_true")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-ok", label="u")
    ax[1].plot(t, y[3], "-ob", label="v")
    ax[1].plot(t, y_true[2], "--xr", label="u_true")
    ax[1].plot(t, y_true[3], "--xg", label="v_true")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(tp, yp[4], "-ok", label="mup")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(tp, yp[5], "-ok", label="lap")
    ax[3].plot(tp, yp_true[5], "--xr", label="lap")
    ax[3].legend()
    ax[3].grid()

    plt.show()
