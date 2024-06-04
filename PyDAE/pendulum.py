import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dae import BDF, Radau, TRBDF2


def make_pendulum(m, g, l, index=3):
    assert index in [1, 2, 3, "GGL"]

    # TODO: Add step size to the rhs that we can scale constraint equations by 
    # h in order to make the Jacobian well behaved for h -> 0. I think this is 
    # important for BDF.
    def F(t, vy, vy_dot, h=1):
        """Cartesian pendulum, see Hairer1996 Section VII Example 2."""
        if index == "GGL":
            # stabilized index 1
            x, y, u, v, _, _ = vy
            x_dot, y_dot, u_dot, v_dot, la, mu = vy_dot
            # # # stabilized index 2
            # x, y, u, v, la, mu = vy
            # x_dot, y_dot, u_dot, v_dot, _, _ = vy_dot

            R = np.zeros(6, dtype=np.common_type(vy, vy_dot))
            R[0] = x_dot - u - 2 * x * mu
            R[1] = y_dot - v - 2 * y * mu
            R[2] = m * u_dot - 2 * x * la
            R[3] = m * v_dot - 2 * y * la + m * g
            R[4] = 2 * x * u + 2 * y * v
            R[5] = x * x + y * y - l * l
            # R[4] /= h
            # R[5] /= h
        else:
            x, y, x_dot, y_dot, _ = vy
            u, v, u_dot, v_dot, la = vy_dot

            R = np.zeros(5, dtype=np.common_type(vy, vy_dot))
            R[0] = u - x_dot
            R[1] = v - y_dot
            R[2] = m * u_dot + 2 * x * la
            R[3] = m * v_dot + 2 * y * la + m * g
            match index:
                case 3:
                    R[4] = x * x + y * y - l * l
                case 2:
                    R[4] = 2 * x * x_dot + 2 * y * y_dot
                case 1:
                    R[4] = 2 * x * u_dot + 2 * y * v_dot + 2 * u * u + 2 * v * v

        return R
    
    def F_y(t, vy, vy_dot):
        assert index == "GGL"

        # stabilized index 1
        x, y, u, v, _, _ = vy
        x_dot, y_dot, u_dot, v_dot, la, mu = vy_dot
        # # # stabilized index 2
        # x, y, u, v, la, mu = vy
        # x_dot, y_dot, u_dot, v_dot, _, _ = vy_dot

        # fmt: off
        J = np.array([
            [-2 * mu,       0,    -1,      0, 0, 0],
            [      0, -2 * mu,     0,     -1, 0, 0],
            [-2 * la,       0,     0,      0, 0, 0],
            [      0, -2 * la,     0,      0, 0, 0],
            [  2 * u,   2 * v, 2 * x,  2 * y, 0, 0],
            [  2 * x,   2 * y,     0,      0, 0, 0],
        ])
        # fmt: on
        return J

    def F_y_dot(t, vy, vy_dot):
        assert index == "GGL"

        # stabilized index 1
        x, y, u, v, _, _ = vy
        x_dot, y_dot, u_dot, v_dot, la, mu = vy_dot
        # # # stabilized index 2
        # x, y, u, v, la, mu = vy
        # x_dot, y_dot, u_dot, v_dot, _, _ = vy_dot

        # fmt: off
        J = np.array([
            [1, 0, 0, 0,      0, -2 * x],
            [0, 1, 0, 0,      0, -2 * y],
            [0, 0, m, 0, -2 * x,      0],
            [0, 0, 0, m, -2 * y,      0],
            [0, 0, 0, 0,      0,      0],
            [0, 0, 0, 0,      0,      0],
        ])
        # fmt: on
        return J
    
    n = 6 if index == "GGL" else 5

    mass_matrix = np.diag(np.concatenate([np.ones(n), np.zeros(n)]))
    
    # Hairer1999 Remark above 5.1
    def rhs(t, z, h=1):
        y = z[:n]
        y_dot = z[n:]
        z_dot = np.zeros_like(z)
        z_dot[:n] = y_dot
        z_dot[n:] = F(t, y, y_dot, h)
        return z_dot
    
    def jac(t, z):
        y = z[:n]
        y_dot = z[n:]

        jac = np.zeros((2 * n, 2 * n))
        jac[:n, n:] = np.eye(n)
        jac[n:, :n] = F_y(t, y, y_dot)
        jac[n:, n:] = F_y_dot(t, y, y_dot)
        return jac

        from cardillo.math.approx_fprime import approx_fprime
        jac_num = approx_fprime(z, lambda z: rhs(t, z), eps=1e-20, method="cs")
        diff = jac - jac_num
        error = np.linalg.norm(diff)
        print(f"error jac: {error}")
        return jac_num
        
    return mass_matrix, rhs, jac


if __name__ == "__main__":
    m = 1
    l = 1
    g = 10
    # index = 2
    # index = 3
    index = "GGL"

    # time span
    t0 = 0
    # t1 = 5e1
    t1 = 10
    t_span = (t0, t1)

    # initial conditions
    if index == "GGL":
        y0 = np.array([l, 0, 0, 0, 0, 0], dtype=float)
        y_dot0 = np.array([0, 0, 0, -g, 0, 0], dtype=float)
        # var_index = np.concatenate((np.zeros(6, dtype=int), np.ones(6, dtype=int)), dtype=int)
        # var_index = np.concatenate((np.zeros(6, dtype=int), 2 * np.ones(6, dtype=int)), dtype=int)
        # works for BDF and Radau
        var_index = np.concatenate((np.zeros(10, dtype=int), np.ones(2, dtype=int)), dtype=int)
        # works for BDF and Radau
        # var_index = np.concatenate((np.zeros(10, dtype=int), 2 * np.ones(2, dtype=int)), dtype=int)
        # # TODO: This works for BDF since it scales the error in Newton for the constraint equations.
        # var_index = np.concatenate((np.zeros(6, dtype=int), np.ones(4, dtype=int), 2 * np.ones(2, dtype=int)), dtype=int)
    else:
        y0 = np.array([l, 0, 0, 0, 0], dtype=float)
        y_dot0 = np.array([0, 0, 0, -g, 0], dtype=float)

        # var_index = np.concatenate((np.zeros(5, dtype=int), index * np.ones(5, dtype=int)), dtype=int)
        # var_index = np.concatenate((np.zeros(7, dtype=int), index * np.ones(3, dtype=int)), dtype=int)
        # np.array([0, 0, 3, 3, 3, 0, 0, 3, 3, 3], dtype=int)
        var_index = np.concatenate((np.zeros(7, dtype=int), index * np.ones(3, dtype=int)), dtype=int)
        # var_index = np.concatenate((np.zeros(7, dtype=int), (index - 1) * np.ones(2, dtype=int), index * np.ones(1, dtype=int)), dtype=int)

    z0 = np.concatenate((y0, y_dot0))

    # solver options
    # rtol = atol = 1e-12
    # rtol = atol = 1e-10
    # rtol = atol = 1e-8
    # rtol = atol = 1e-6
    rtol = atol = 1e-5
    # rtol = atol = 1e-4
    # rtol = atol = 1e-3
    # rtol = atol = 1e-2
    # atol = 1e-8
    # rtol = 1e-8
    # atol = 1e-6
    # rtol = 1e-6
    # atol = 1e-5
    # rtol = 1e-5

    # # reference solution
    # mass_matrix, rhs = make_pendulum(DAE=False)
    # sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    # t_scipy = sol.t
    # y_scipy = sol.y

    # dae solution
    mass_matrix, rhs, jac = make_pendulum(m, g, l, index=index)
    method = Radau
    # method = BDF
    # method = TRBDF2
    start = time.time()
    sol = solve_ivp(rhs, t_span, z0, jac=jac, atol=atol, rtol=rtol, method=method, mass_matrix=mass_matrix, var_index=var_index)
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    z = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"success: {success}")
    print(f"status: {status}")
    print(f"message: {message}")

    # y = z[:5]
    # y_dot = z[5:]
    y = z

    # from dae.euler import euler
    # rtol = 1e-3
    # atol = 1e-6
    # t, y = euler(rhs, z0, t_span, rtol, atol, mass_matrix, var_index)
    # y = y.T

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y[0], "-ok", label="x")
    ax[0].plot(t, y[1], "--xk", label="y")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-ok", label="u")
    ax[1].plot(t, y[3], "--xk", label="v")
    # ax[1].plot(t, y_dot[0], "-.r", label="x_dot")
    # ax[1].plot(t, y_dot[1], ":r", label="u_dot")
    ax[1].legend()
    ax[1].grid()

    # ax[2].plot(t, y_dot[2], "-k", label="u_dot")
    # ax[2].plot(t, y_dot[3], "--k", label="v_dot")
    # ax[2].legend()
    # ax[2].grid()

    # ax[3].plot(t, y[4], "-k", label="la dt")
    # ax[3].plot(t, y_dot[4], "-k", label="la")
    ax[3].plot(t, y[10], "-ok", label="la")
    ax[3].plot(t, y[11], "--xk", label="mu")
    ax[3].legend()
    ax[3].grid()

    plt.show()
