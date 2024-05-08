import numpy as np

from scipy.integrate import solve_ivp
from scipy.sparse import eye
from scipy.optimize._numdiff import approx_derivative
from dae import BDF, Radau
from numpy.testing import assert_allclose


def generate_Jay1993():

    # construct singular mass matrix
    mass_matrix = eye(3, format="csr")
    mass_matrix[-1, -1] = 0

    # DAE index
    # var_index = [0, 2, 2]
    var_index = [0, 0, 2]

    # initial conditions
    y0 = np.ones(3, dtype=float)

    # tolerances and t_span
    rtol = atol = 1.0e-8
    # t_span = (0, 20)
    t_span = (0, 2)

    def fun(t, y):
        """Index2 DAE found in Jay1993 Example 7.

        References:
        -----------
        Jay1993: https://link.springer.com/article/10.1007/BF01990349
        """
        y1, y2, z = y

        dy = np.array(
            [
                y1 * y2**2 * z**2,
                y1**2 * y2**2 - 3 * y2**2 * z,
                y1**2 * y2 - 1.0,
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
        ax.plot(t, y[2], "--b", label="z")

        ax.plot(t, np.exp(t), "-r", label="y1 true")
        ax.plot(t, np.exp(-2 * t), "-g", label="y2 true")
        ax.plot(t, np.exp(2 * t), "-b", label="z true")

        ax.grid()
        ax.legend()

        plt.show()

    def errors(t, y):
        assert_allclose(y[0], np.exp(t), rtol=1e-5)
        assert_allclose(y[1], np.exp(-2 * t), rtol=1e-5)
        assert_allclose(y[2], np.exp(2 * t), rtol=1e-3)
        # dt = t[1] - t[0]
        # error_y1 = np.linalg.norm((y[0] - np.exp(t)) * dt)
        # error_y2 = np.linalg.norm((y[1] - np.exp(-2 * t)) * dt)
        # error_y3 = np.linalg.norm((y[2] - np.exp(2 * t))  * dt)
        # print(f"error: [{error_y1}, {error_y2}, {error_y3}]")
        # return error_y1, error_y2, error_y3

    return y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot, errors


if __name__ == "__main__":
    y0, mass_matrix, var_index, fun, jac, rtol, atol, t_span, plot, errors = generate_Jay1993()

    method = BDF
    # method = Radau
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

    errors(sol.t, sol.y)
    plot(sol.t, sol.y)
