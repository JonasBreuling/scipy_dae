import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.testing import assert_allclose
from PyDAE.integrate._dae.dae import solve_dae, RadauDAE, BDFDAE
from PyDAE.integrate._dae.common import consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative


"""Index3 DAE found in Jay1995 Example 7.1 and 7.2.

References:
-----------
Jay1995: https://doi.org/10.1016/0168-9274(95)00013-K
"""
def F(t, y, yp, nonlinear_multiplier=False):
    y1, y2, z1, z2, _ = y
    y1p, y2p, z1p, z2p, lap = yp

    if nonlinear_multiplier:
        dy3 = -y1 * y2**2 * z2**3 * lap**2
    else:
        dy3 = -y1 * y2**2 * z2**2 * lap

    F = np.zeros(5, dtype=y.dtype)
    F[0] = y1p - (2 * y1 * y2 * z1 * z2)
    F[1] = y2p - (-y1 * y2 * z2**2)
    F[2] = z1p - ((y1 * y2 + z1 * z2) * lap)
    if nonlinear_multiplier:
        F[3] = z2p - (-y1 * y2**2 * z2**3 * lap**2)
    else:
        F[3] = z2p - (-y1 * y2**2 * z2**2 * lap)
    F[4] = y1 * y2**2 - 1.0

    return F


def jac(t, y, yp, f):
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


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 3
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-5

    # initial conditions
    y0 = np.array([1, 1, 1, 1, 0], dtype=float)
    yp0 = np.array([2, -1, 2, -1, 1], dtype=float)

    # F0 = F(t0, y0, yp0)
    # print(f"F0: {F0}")
    # exit()

    # # TODO: This seems to be wrong here!
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # y0, yp0, fnorm = consistent_initial_conditions(F, jac, t0, y0, yp0, fixed_y0=None, fixed_yp0=None)
    # print(f"y0: {y0}")
    # print(f"yp0: {yp0}")
    # print(f"fnorm: {fnorm}")
    # exit()

    ##############
    # dae solution
    ##############
    # method = RadauDAE
    method = BDFDAE
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, max_order=3)
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

    # errors
    # assert_allclose(y[0], np.exp(2 * t), rtol=1e-6)
    # assert_allclose(y[1], np.exp(-t), rtol=1e-6)
    # assert_allclose(y[2], np.exp(2 * t), rtol=1e-6)
    # assert_allclose(y[3], np.exp(-t), rtol=1e-6)
    # assert_allclose(y[4], np.exp(t), rtol=1e-6)
    dt = t[1] - t[0]
    error_y1 = np.linalg.norm((y[0] - np.exp(2 * t)) * dt)
    error_y2 = np.linalg.norm((y[1] - np.exp(-t)) * dt)
    error_z1 = np.linalg.norm((y[2] - np.exp(2 * t))  * dt)
    error_z2 = np.linalg.norm((y[3] - np.exp(-t))  * dt)
    error_lap = np.linalg.norm((yp[4] - np.exp(tp))  * dt)
    print(f"error: [{error_y1}, {error_y2}, {error_z1}, {error_z2}, , {error_lap}]")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, y[1], "--g", label="y2")
    ax.plot(t, y[2], "-.r", label="z1")
    ax.plot(t, y[3], "-.g", label="z2")
    ax.plot(tp, yp[4], "--b", label="u")

    ax.plot(t, np.exp(2 * t), "-r", label="y1/z1 true")
    ax.plot(t, np.exp(-t), "-g", label="y2/z2 true")
    ax.plot(t, np.exp(t), "-b", label="u true")

    ax.grid()
    ax.legend()

    plt.show()
