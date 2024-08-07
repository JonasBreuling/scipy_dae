import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.integrate import solve_ivp


def smoothstep2(x, x_min=0, x_max=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


EPS = 1e-6

t0 = 0
t1 = 10

# geometry of the rod
length = 1 # [m]
# width = 1e-2 # [m]
width = 1e-3 # [m]
density = 8.e3  # [kg / m^3]

# material properties
E = 260.0e9 # [N / m^2]
G = 100.0e9 # [N / m^2]
# E = 260.0e6 # [N / m^2]
# G = 100.0e6 # [N / m^2]


# # [m] -> [mm]
# length *= 1e3
# # cross section properties
# width *= 1e3
# density *= 1e-9

# # [s] -> [ms]
# t1 *= 1e3

# # [N / m^2] -> [kN / mm^2]
# E *= 1e-9
# G *= 1e-9

A = width**2
I = width**4 / 12

EA = E * A
GA = G * A

M = np.diag([A, A, I]) * density * length**3 / 3


def A_IB(phi):
    sphi, cphi = np.sin(phi), np.cos(phi)
    return np.array([
        [cphi, -sphi],
        [sphi,  cphi],
    ])

def df_int(xi, vq):
    x, y, phi = vq

    A_IB_ = A_IB(phi * xi)
    n = A_IB_ @ np.diag([E * A, G * A]) @ (
        A_IB_.T @ vq[:2] / length - np.array([1, 0])
    )
    m = E * I * phi / length

    K1 = np.array([
        [1,  0, 0],
        [0,  1, 0],
        [y, -x, 1]
    ])

    return K1 @ np.array([*n, m])


def f_int(vq):
    # one point quadrature
    x = np.array([0])
    w = np.array([2])
    
    # # two point quadrature
    # x = np.array([-np.sqrt(1 / 3), np.sqrt(1 / 3)])
    # w = np.array([1, 1])

    # transform from [-1, 1] on [0, 1]
    b = 1
    a = 0
    w = (b - a) / 2 * w
    x = (a + b) / 2 + (b - a) / 2 * x

    return np.sum([df_int(xi, vq) * wi for (xi, wi) in zip(x, w)], axis=0)

    # # exact internal forces
    # x, y, phi = vq

    # K1 = np.array([
    #     [1,  0, 0],
    #     [0,  1, 0],
    #     [y, -x, 1]
    # ])

    # sphi, cphi = np.sin(phi), np.cos(phi)

    # if abs(phi) > EPS:
    #     cphi2 = 0.5 + sphi * cphi / (2 * phi)
    #     sphi2 = 0.5 - np.sin(2 * phi) / (4 * phi)
    #     sphi_cphi = sphi**2 / (2 * phi)

    #     n0 = EA * np.array([
    #         sphi,
    #         1 - cphi
    #     ]) / phi
    # else:
    #     n0 = EA * np.array([
    #         # https://www.wolframalpha.com/input?i=taylor+series+sin%28x%29+%2F+x
    #         1 - phi**2 / 6 + phi**4 / 120,
    #         # https://www.wolframalpha.com/input?i=taylor+series+%281+-+cos%28x%29%29+%2F+x
    #         phi / 2 - phi**3 / 24 + phi**5 / 720
    #     ])

    # K_phi = np.array([
    #     [EA * cphi**2 + GA * sphi**2,     (EA - GA) * cphi * sphi],
    #     [    (EA - GA) * cphi * sphi, EA * sphi**2 + GA * cphi**2],
    # ])

    # n = K_phi @ vq[:2] - n0
    # m = E * I * phi / length

    # return K1 @ np.array([*n, m])


def m(t):
    m_max = E * I / length * 2
    return m_max * (
        smoothstep2(t, 0, 4)
        - smoothstep2(t, 4, 4.1)
    )
    # if t < 4:
    #     return t * m_max
    # else:
    #     return 0.0

def f_ext(t, vq):
    return np.array([
        0,
        0,
        m(t)
    ])


def F(t, vy, vyp):
    vq, vu = vy[:3], vy[3:]
    vqp, vup = vyp[:3], vyp[3:]

    return np.concatenate([
        vqp - vu,
        M @ vup + f_int(vq) - f_ext(t, vq),
    ])

def f(t, vy):
    vq, vu = vy[:3], vy[3:]

    q_dot = vu
    u_dot = -np.linalg.solve(M, f_int(vq) - f_ext(t, vq))

    return np.concatenate([q_dot, u_dot])


if __name__ == "__main__":
    # time span
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e4))
    t_eval = None

    # method = "BDF"
    method = "Radau"
    # method = "RK23"

    # initial positions
    q0 = np.array([length, 0, 0], dtype=float)

    # r_OP1 = np.array([0.76738219, 0.54021608]) * 1e3
    # phi1 = 1.22664684
    # q0 = np.array([*r_OP1, phi1])

    # initial velocities
    u0 = np.array([0, 0, 0], dtype=float)

    y0 = np.concatenate([q0, u0])

    q_dot0 = u0
    u_dot0 = np.array([0, 0, 0], dtype=float)
    yp0 = np.concatenate([q_dot0, u_dot0])

    F0 = F(t0, y0, yp0)
    print(f"F0: {F0}")

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, F0 = consistent_initial_conditions(F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    # solver options
    atol = rtol = 1e-3

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, stages=5)
    # sol = solve_ivp(f, t_span, y0, method=method, t_eval=t_eval, atol=atol, rtol=rtol)
    end = time.time()
    t = sol.t
    y = sol.y
    success = sol.success
    status = sol.status
    message = sol.message
    print(f"message: {message}")
    print(f"elapsed time: {end - start}")
    print(f"nfev: {sol.nfev}")
    print(f"njev: {sol.njev}")
    print(f"nlu: {sol.nlu}")

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].set_xlabel("t")
    ax[0].set_xlabel("x")
    ax[0].grid()
    ax[0].plot(t, y[0], "-ok")

    ax[1].set_xlabel("t")
    ax[1].set_xlabel("y")
    ax[1].grid()
    ax[1].plot(t, y[1], "-ok")

    ax[2].set_xlabel("t")
    ax[2].set_xlabel("phi")
    ax[2].grid()
    ax[2].plot(t, y[2], "-ok")

    ax[3].set_xlabel("t")
    ax[3].set_xlabel("h")
    ax[3].grid()
    ax[3].plot(t[1:], np.diff(t), "-ok")

    plt.show()
