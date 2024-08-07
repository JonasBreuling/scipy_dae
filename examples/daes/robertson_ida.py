import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
# from diffeqpy import de


"""Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

References:
-----------
mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011 \\
Sundials IDA (page 6): https://computing.llnl.gov/sites/default/files/ida_examples-5.7.0.pdf
"""

def f(t, y):
    y1, y2, y3 = y

    yp = np.zeros(3, dtype=float)
    yp[0] = -0.04 * y1 + 1e4 * y2 * y3
    yp[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
    yp[2] = 3e7 * y2**2

    return yp

def F(t, y, yp):
    # return yp - f(t, y)
    y1, y2, y3 = y
    y1p, y2p, y3p = yp

    F = np.zeros(3, dtype=float)
    F[0] = y1p - (-0.04 * y1 + 1e4 * y2 * y3)
    F[1] = y2p - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2)
    F[2] = y1 + y2 + y3 - 1

    return F

def jac(t, y, yp):
    y1, y2, y3 = y

    Jyp = np.diag([1, 1, 0])
    Jy = np.array([
        [0.04, -1e4 * y3, -1e4 * y2],
        [-0.04, 1e4 * y3 + 6e7 * y2, 1e4 * y2],
        [1, 1, 1],
    ])
    return Jy, Jyp


if __name__ == "__main__":
    # time span
    t0 = 0
    t1 = 1e7
    t1 = 4e10
    t_span = (t0, t1)
    t_eval = np.logspace(-6, 7, num=1000)
    # t_eval = None

    # method = "BDF"
    method = "Radau"

    # initial conditions
    y0 = np.array([1, 0, 0], dtype=float)
    yp0 = f(t0, y0)

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, fnorm = consistent_initial_conditions(F, t0, y0, yp0, fixed_y0=[0, 1])
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"fnorm: {fnorm}")

    # solver options
    # atol = rtol = 1e-6
    atol = rtol = 1e-8
    # rtol = 1e-4
    # atol = [1e-8, 1e-6, 1e-6]
    # rtol = 1e-4
    # atol = 1e-6

    ####################
    # reference solution
    ####################
    start = time.time()
    sol = solve_ivp(f, t_span, y0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t_scipy = sol.t
    y_scipy = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    ##############
    # dae solution
    ##############
    # jac = None
    stages = 7
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, jac=jac, stages=stages)
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

    # #################################
    # # Julia solution (Use IDA solver)
    # #################################
    # differential_vars = [True, True, False]
    # y0 = [*y0]
    # yp0 = [*yp0]
    # prob = de.DAEProblem(
    #     lambda yp, y, p, t: F(t, y, yp), 
    #     yp0, 
    #     y0, 
    #     t_span, 
    #     differential_vars=differential_vars,
    #     abstol=atol, 
    #     reltol=rtol,
    # )
    # # see https://docs.sciml.ai/DiffEqDocs/stable/api/sundials/#sundials for options
    # solver = de.IDA()
    # start = time.time()
    # sol = de.solve(prob, solver)
    # end = time.time()
    # print(f"elapsed time: {end - start}")
    # print(f"Number of function 1 evaluations: {sol.stats.nf}")
    # print(f"Number of function 2 evaluations: {sol.stats.nf2}")
    # print(f"Number of W matrix evaluations: {sol.stats.nw}")
    # print(f"Number of Jacobians created: {sol.stats.njacs}")
    # print(f"Number of nonlinear solver iterations: {sol.stats.nnonliniter}")
    # print(f"Number of nonlinear convergence failuers: {sol.stats.nnonlinconvfail}")
    # print(f"Number of accepted steps: {sol.stats.naccept}")
    # print(f"Number of rejected steps: {sol.stats.nreject}")

    # t_IDA = np.array(sol.t)
    # y_IDA = np.array([ui for ui in sol.u]).T

    # visualization
    fig, ax = plt.subplots()

    ax.plot(t, y[0], "-ok", label="y1 DAE" + f" ({method})", mfc="none")
    ax.plot(t, y[1] * 1e4, "-ob", label="y2 DAE" + f" ({method})", mfc="none")
    ax.plot(t, y[2], "-og", label="y3 DAE" + f" ({method})", mfc="none")

    ax.plot(t_scipy, y_scipy[0], "xr", label="y1 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[1] * 1e4, "xy", label="y2 scipy Radau", markersize=7)
    ax.plot(t_scipy, y_scipy[2], "xm", label="y3 scipy Radau", markersize=7)
    
    # ax.plot(t_IDA, y_IDA[0], "sr", label="y1 IDA", markersize=7)
    # ax.plot(t_IDA, y_IDA[1] * 1e4, "sy", label="y2 IDA", markersize=7)
    # ax.plot(t_IDA, y_IDA[2], "sm", label="y3 IDA", markersize=7)

    ax.set_xlabel("t")
    ax.set_xscale("log")
    ax.legend()
    ax.grid()

    plt.show()