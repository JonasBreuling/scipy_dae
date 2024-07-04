import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.optimize._numdiff import approx_derivative


"""Index2 DAE found in Jay1993 Example 7.

References:
-----------
Jay1993: https://link.springer.com/article/10.1007/BF01990349
"""
def F(t, y, yp):
    y1, y2, _ = y
    y1p, y2p, zp = yp

    F = np.zeros(3, dtype=y.dtype)
    # TODO: This multiplier enters nonlinear. Can we cope with that?
    F[0] = y1p - (y1 * y2**2 * zp**2)
    # F[0] = y1p - (y1 * y2**2 * zp)
    F[1] = y2p - (y1**2 * y2**2 - 3 * y2**2 * zp)
    F[2] = y1**2 * y2 - 1.0

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

jac = None


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 2
    t_span = (t0, t1)

    # tolerances
    rtol = atol = 1e-4

    # initial conditions
    y0 = np.array([1, 1, 0], dtype=float)
    # yp0 = np.array([0, 0, 1], dtype=float) # TODO: Why is this not consistent?
    yp0 = np.array([1, -2, 1], dtype=float)
    # print(F(t0, y0, yp0))
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
    start = time.time()
    # sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method="Radau", stages=1)
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method="BDF")
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
    # assert_allclose(y[0], np.exp(t), rtol=1e-8)
    # assert_allclose(y[1], np.exp(-2 * t), rtol=1e-8)
    # assert_allclose(y[2], np.exp(2 * t), rtol=1e-8)
    dt = t[1] - t[0]
    error_y1 = np.linalg.norm((y[0] - np.exp(t)) * dt)
    error_y2 = np.linalg.norm((y[1] - np.exp(-2 * t)) * dt)
    # Note: We have an expected order reduction of h here since we control the
    #       error of y[2] instead of yp[2].
    # error_y3 = np.linalg.norm((yp[2] - np.exp(2 * tp)) * dt)
    error_y3 = np.linalg.norm((yp[2] - np.exp(2 * tp)) * dt**2)
    print(f"error: [{error_y1}, {error_y2}, {error_y3}]")

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "--r", label="y1")
    ax.plot(t, y[1], "--g", label="y2")
    ax.plot(tp, yp[2], "--b", label="z")

    ax.plot(t, np.exp(t), "-r", label="y1 true")
    ax.plot(t, np.exp(-2 * t), "-g", label="y2 true")
    ax.plot(t, np.exp(2 * t), "-b", label="z true")

    ax.grid()
    ax.legend()

    plt.show()
