import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.sparse import eye
from scipy.optimize._numdiff import approx_derivative
from dae import BDF, Radau, TRBDF2
from numpy.testing import assert_allclose


def generate_bouncing_ball(redundant):
    gravity = 10

    def fun(t, y, y0):
        y, u, la = y
        y0, u0, _ = y0
        r = 1e-1
        eps = 0.25

        prox_arg_N = r * y - la
        if prox_arg_N <= 0:
            prox_N = y
            xi_N = u + eps * u0
            prox_arg_N_dot = r * xi_N - la
            if prox_arg_N_dot <= 0:
                prox_N_dot = xi_N
            else:
                prox_N_dot = la
        else:
            prox_N = la
            prox_N_dot = la
        # prox_N = min(la, y)
        # prox_N_dot = (y0 <= 0) * min(la, u + eps * u0)

        # if y <= 0:
        #     prox_N_dot = min(la, u + eps * u0) * min(la, y)
        # else:
        #     prox_N_dot = min(la, y)
        # prox_N_dot = min(min(la, y), (y0 <= 0) * min(la, u + eps * u0))
            # prox_N_dot = min(la, y)
        # prox_N_dot = min(la, y) * min(la, u + eps * u0)
        # prox_N_dot = (mu > 0) * min(la, u + eps * u0)
        # prox_N_dot = (mu > 0) * min(la, u + eps * u0)

        # g = np.array([prox_N_dot])
        g = np.array(
            [
                prox_N,
                prox_N_dot,
            ]
        )

        return np.concatenate(
            (
                [
                    u,
                    -gravity + la,
                ],
                g,
            )
        )

    # the jacobian is computed via finite-differences
    def jac(t, x):
        return approx_derivative(
            fun=lambda x: fun(t, x), x0=x, method="3-point", rel_step=1e-6
        )

    def plot(t, y):
        fig, ax = plt.subplots()

        ax.plot(t, y[0], "-k", label="y")
        ax.plot(t, y[1], "-b", label="u")
        # ax.plot(t, y[2], "-r", label="la")

        ax.grid()
        ax.legend()

        plt.show()

    # construct singular mass matrix
    mass_matrix = eye(3, format="csr")
    mass_matrix[-1, -1] = 0

    # DAE index
    var_index = [0, 3, 3]

    # initial conditions
    y0 = np.array([1, 0, 0])

    # tolerances and t_span
    rtol = atol = 1.0e-12
    t_span = (0, 2)

    return y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot

from scipy.optimize import least_squares
def solve(fun, jac, mass_matrix, y0, tspan, dt):
    t = np.arange(tspan[0], tspan[1] + dt, dt)
    nt = len(t)
    ny = y0.shape[0]
    sol_y = np.empty((nt, ny))

    for i, ti in enumerate(t):
        def R(y):
            R = -dt * fun(ti, y, y0)
            R[:2] += (y - y0)[:2]
            return R
        # method = "lm"
        method = "trf"
        sol = least_squares(R, y0, verbose=True, method=method, jac="3-point")
        y1 = sol.x
        
        sol_y[i] = y1
        y0 = y1.copy()

    return t, sol_y

if __name__ == "__main__":
    # redundant = False
    redundant = True
    y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot = (
        generate_bouncing_ball(redundant=redundant)
    )

    dt = 5e-3
    t, sol_y = solve(fun, jac, mass_matrix, y0, t_span, dt)

    fig, ax = plt.subplots()
    ax.plot(t, sol_y[:, 0], "-k", label="y")
    ax.plot(t, sol_y[:, 1], "--b", label="u")
    ax.plot(t, dt * sol_y[:, 2], "-.r", label="la")
    ax.grid()
    ax.legend()
    plt.show()

    exit()

    method = BDF
    # method = Radau
    # method = TRBDF2
    sol = solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        rtol=rtol,
        atol=atol,
        jac=jac,
        method=method,
        mass_matrix=mass_matrix,
        var_index=var_index,
    )
    print(f"sol: {sol}")

    nfev = sol.nfev
    njev = sol.njev
    nlu = sol.nlu
    success = sol.success
    # assert success

    plot(sol.t, sol.y)
