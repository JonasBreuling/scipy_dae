import numpy as np

from scipy.integrate import solve_ivp
from scipy.sparse import eye
from scipy.optimize._numdiff import approx_derivative
from dae import BDF, Radau, TRBDF2
from numpy.testing import assert_allclose


def generate_Jay1995(nonlinear_multiplier):
    def fun(t, y):
        """Index3 DAE found in Jay1995 Example 7.1 and 7.2.

        References:
        -----------
        Jay1995: https://doi.org/10.1016/0168-9274(95)00013-K
        """
        y1, y2, z1, z2, la = y

        if nonlinear_multiplier:
            dy3 = -y1 * y2**2 * z2**3 * la**2
        else:
            dy3 = -y1 * y2**2 * z2**2 * la

        dy = np.array(
            [
                2 * y1 * y2 * z1 * z2,
                -y1 * y2 * z2**2,
                (y1 * y2 + z1 * z2) * la,
                dy3,
                y1 * y2**2 - 1.0,
            ],
            dtype=y.dtype,
        )

        return dy

    # the jacobian is computed via finite-differences
    def jac(t, x):
        return approx_derivative(
            fun=lambda x: fun(t, x), x0=x, method="cs", rel_step=1e-50
        )

    def plot(t, y):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.plot(t, y[0], "--r", label="y1")
        ax.plot(t, y[1], "--g", label="y2")
        ax.plot(t, y[2], "-.r", label="z1")
        ax.plot(t, y[3], "-.g", label="z2")
        ax.plot(t, y[4], "--b", label="u")

        ax.plot(t, np.exp(2 * t), "-r", label="y1/z1 true")
        ax.plot(t, np.exp(-t), "-g", label="y2/z2 true")
        ax.plot(t, np.exp(t), "-b", label="u true")

        ax.grid()
        ax.legend()

        plt.show()

    def errors(t, y):
        # rtol = 1
        assert_allclose(y[0], np.exp(2 * t), rtol=1e-6)
        assert_allclose(y[1], np.exp(-t), rtol=1e-6)
        assert_allclose(y[2], np.exp(2 * t), rtol=1e-6)
        assert_allclose(y[3], np.exp(-t), rtol=1e-6)
        assert_allclose(y[4], np.exp(t), rtol=1e-3)
        # dt = t[1] - t[0]
        # error_y1 = np.linalg.norm((y[0] - np.exp(2 * t)) * dt)
        # error_y2 = np.linalg.norm((y[1] - np.exp(-t)) * dt)
        # error_z1 = np.linalg.norm((y[2] - np.exp(2 * t))  * dt)
        # error_z2 = np.linalg.norm((y[3] - np.exp(-t))  * dt)
        # error_la = np.linalg.norm((y[4] - np.exp(t))  * dt)
        # print(f"error: [{error_y1}, {error_y2}, {error_z1}, {error_z2}, , {error_la}]")
        # # return error_y1, error_y2, error_z1, error_z2, error_la

    # construct singular mass matrix
    mass_matrix = eye(5, format="csr")
    mass_matrix[-1, -1] = 0

    # DAE index
    var_index = [0, 0, 2, 2, 3]

    # initial conditions
    y0 = np.ones(5, dtype=float)

    # tolerances and t_span
    # rtol = atol = 1.0e-8
    rtol = atol = 1.0e-5
    # t_span = (0, 20)
    t_span = (0, 1)

    return y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot, errors


if __name__ == "__main__":
    nonlinear_multiplier = False
    # nonlinear_multiplier = True
    y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot, errors = generate_Jay1995(
        nonlinear_multiplier=nonlinear_multiplier,
    )

    # method = BDF
    # method = Radau
    method = TRBDF2
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
    assert success

    # errors(sol.t, sol.y)
    plot(sol.t, sol.y)
