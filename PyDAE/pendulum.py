import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dae import BDF, Radau

def make_pendulum(m, g, l, index=3):
    assert index in [1, 2, 3]

    def F(t, vy, vy_dot):
        """Cartesian pendulum, see Hairer1996 Section VII Example 2."""
        x, y, x_dot, y_dot, _ = vy
        u, v, u_dot, v_dot, la = vy_dot
        # TODO: Why is this so bad?
        # x, y, x_dot, y_dot, la = vy
        # u, v, u_dot, v_dot, _ = vy_dot

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
    
    mass_matrix = np.diag(np.concatenate([np.ones(5), np.zeros(5)]))
    
    # Hairer1999 Remark above 5.1
    def rhs(t, z):
        y = z[:5]
        y_dot = z[5:]
        z_dot = np.zeros_like(z)
        z_dot[:5] = y_dot
        z_dot[5:] = F(t, y, y_dot)
        return z_dot
    
    # mass_matrix = np.eye(5)
    # if index > 0:
    #     mass_matrix[-1, -1] = 0
    
    # def rhs(t, vy):
    #     """Cartesian pendulum, see Hairer1996 Section VII Example 2."""
    #     x, y, x_dot, y_dot, la = vy

    #     vy_dot = np.zeros(5, dtype=vy.dtype)
    #     vy_dot[0] = x_dot
    #     vy_dot[1] = y_dot
    #     vy_dot[2] = -2 * x * la / m
    #     vy_dot[3] = -2 * y * la / m - g
    #     match index:
    #         case 3:
    #             vy_dot[4] = x * x + y * y - l * l
    #         case 2:
    #             vy_dot[4] = 2 * x * x_dot + 2 * y * y_dot
    #         case 1:
    #             vy_dot[4] = 2 * la * (x ** 2 + y ** 2) / m  + g * y - (x_dot ** 2 + y_dot ** 2)
    #         case default:
    #             raise NotImplementedError

    #     return vy_dot
    
    return mass_matrix, rhs


if __name__ == "__main__":
    m = 1
    l = 1
    g = 10
    index = 2

    # time span
    t0 = 0
    t1 = 5e1
    t_span = (t0, t1)

    # initial conditions
    y0 = np.array([l, 0, 0, 0, 0], dtype=float)
    # z0 = y0
    y_dot0 = np.array([0, 0, 0, -g, 0], dtype=float)
    z0 = np.concatenate((y0, y_dot0))
    # var_index = np.concatenate((np.zeros(5, dtype=int), index * np.ones(5, dtype=int)), dtype=int)
    var_index = np.array([0, 0, 0, 0, 0, 2, 2, 2, 2, 2], dtype=int)
    # var_index = np.array([0, 0, 3, 3, 3, 3, 3, 3, 3, 3], dtype=int)

    # solver options
    # atol = 1e-6
    # rtol = 1e-6
    atol = 1e-5
    rtol = 1e-5

    # # reference solution
    # mass_matrix, rhs = make_pendulum(DAE=False)
    # sol = solve_ivp(rhs, t_span, y0, atol=atol, rtol=rtol, method="Radau")
    # t_scipy = sol.t
    # y_scipy = sol.y

    # dae solution
    mass_matrix, rhs = make_pendulum(m, g, l, index=index)
    # sol = solve_ivp(rhs, t_span, z0, atol=atol, rtol=rtol, method=BDF, mass_matrix=mass_matrix, var_index=var_index)
    sol = solve_ivp(rhs, t_span, z0, atol=atol, rtol=rtol, method=Radau, mass_matrix=mass_matrix, var_index=var_index)
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

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y[0], "-k", label="x")
    ax[0].plot(t, y[1], "--k", label="y")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, y[2], "-k", label="u")
    ax[1].plot(t, y[3], "--k", label="v")
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
    ax[3].plot(t, y[4], "-k", label="la")
    ax[3].legend()
    ax[3].grid()

    plt.show()
