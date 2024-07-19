import time
import numpy as np
import matplotlib.pyplot as plt
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.integrate import solve_ivp

# EPS = 1e-6
EPS = 0
# EPS = np.infty

t0 = 0
t1 = 5

# geometry of the rod
length = 1 # [m]
# width = 0.01 # [m]
width = 1e-3 # [m]
density = 8.e3  # [kg / m^3]

# material properties
E = 260.0e9 # [N / m^2]
G = 100.0e9 # [N / m^2]


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

M = np.diag([A, A, I]) * density * length**3 / 3

np.set_printoptions(3)
print(f"M:\n{M}")


def gamma(vq):
    x, y, phi = vq
    aphi = abs(phi)
    if aphi > EPS:
        g = 0.5 * aphi * np.cos(aphi / 2) / np.sin(aphi / 2)
    else:
        # https://www.wolframalpha.com/input?i=taylor+series+abs%28x%29+cot%28abs%28x%29+%2F+2%29+%2F+2
        g = 1 - phi**2 / 12 - phi**4 / 720

    return np.array([
        [       g, phi / 2],
        [-phi / 2,       g]
    ]) @ np.array([
        x,
        y,
    ]) / length


def W(vq):
    x, y, phi = vq
    sphi, cphi = np.sin(phi * length), np.cos(phi * length)

    ga_x, ga_y = gamma(vq)

    if abs(phi) > EPS:
        diag = sphi / phi
        off_diag = (1 - cphi) / phi
    else:
        # https://www.wolframalpha.com/input?i=taylor+series+sin%28c+*+x%29+%2F+x
        diag = length - length**3 * phi**2 / 6 + length**5 * phi**4 / 120

        # https://www.wolframalpha.com/input?i=taylor+series+%281+-+cos%28c+*+x%29%29+%2F+x
        off_diag = length**2 * phi / 2 - length**4 * phi**3 / 24 + length**6 * phi**5 / 720
    
    return np.array([
        [                  diag,               -off_diag, 0],
        [              off_diag,                    diag, 0],
        [0.5 * length**2 * ga_y, -0.5 * length**2 * ga_x, 1],
    ])


def la(vq):
    x, y, phi = vq
    ga_x, ga_y = gamma(vq)

    return np.array([
        E * A * (ga_x - 1),
        G * A * ga_y,
        E * I / length * phi
    ])


def m(t):
    m_max = E * I / length * 0.5
    if t < 2:
        return t * m_max
    else:
        return 0.0

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
        M @ vup + W(vq) @ la(vq) - f_ext(t, vq),
    ])

def f(t, vy):
    vq, vu = vy[:3], vy[3:]

    q_dot = vu
    u_dot = -np.linalg.solve(M, W(vq) @ la(vq) - f_ext(t, vq))

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
    # atol = rtol = 1e-2
    # atol = 1e0
    # rtol = 1e-4
    # atol = rtol = np.array([1e3, 1e3, 1e-3, 1e3, 1e3, 1e-3])
    # atol = rtol = np.array([1e-3, 1e-3, 1e-3, 1e3, 1e3, 1e3])
    # atol = rtol = np.array([1e-1, 1e-1, 1e-1, 1e3, 1e3, 1e3])
    atol = rtol = 1e-3
    # rtol = 1e-3
    # # atol = np.array([1e-3, 1e-3, 1e-3, 1e3, 1e3, 1e3])
    # atol = np.array([1e-3, 1e-3, 1e-3, 1e2, 1e2, 1e2])

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval)
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
    fig, ax = plt.subplots(3, 1)

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

    plt.show()
