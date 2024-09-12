import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.optimize._numdiff import approx_derivative
from scipy_dae.integrate import solve_dae, consistent_initial_conditions
from scipy.integrate import solve_ivp
from scipy.linalg import eig, cdf2rdf
from scipy.optimize import OptimizeResult


EPS = 1e-6
# EPS = 0
# EPS = np.infty

STEEL = True
# STEEL = False


def EulerForward(f, y0, t_span, h):
    t_sol = []
    y_sol = []
    y = y0.copy()
    t0, t1 = t_span
    t = t0
    while t <= t1:
        t += h
        y += h * f(t, y)
        print(f"t: {t}")

        t_sol.append(t)
        y_sol.append(y.copy())

    return OptimizeResult(
        t=np.array(t_sol), 
        y=np.array(y_sol).T,
    )


def EulerBackward(f, y0, t_span, h, eps=1e-6):
    t_sol = []
    y_sol = []
    y = y0.copy()
    t0, t1 = t_span
    t = t0

    while t <= t1:
        t += h
        print(f"t: {t}")

        def residual(y_new):
            return y_new - y - h * f(t, y_new)
        
        def jac(y_new):
            return approx_derivative(residual, y_new)
        
        y_new = y.copy()
        res = residual(y_new)
        error = np.linalg.norm(res)
        while error >= eps:
            y_new -= np.linalg.solve(jac(y_new), res)
            res = residual(y_new)
            error = np.linalg.norm(res)

        y = y_new.copy()

        t_sol.append(t)
        y_sol.append(y.copy())

    return OptimizeResult(
        t=np.array(t_sol), 
        y=np.array(y_sol).T,
    )


def RadauIIA(f, y0, t_span, h, s=3, eps=1e-5):
    t_sol = []
    y_sol = []
    y = y0.copy()
    t0, t1 = t_span
    t = t0

    if s == 1:
        c = np.array([1])
        A = np.array([
            [1.0],
        ])
    elif s == 2:
        A = np.array([
            [5 / 12, -1 / 12],
            [ 3 / 4,   1 / 4],
        ])
    elif s == 3:
        S6 = np.sqrt(6)
        A = np.array([
            [11 / 45 - 7 * S6 / 360, 37 / 225 - 169 * S6 / 1800, -2 / 255 + S6 / 75],
            [37 / 225 + 169 * S6 / 1800, 11 / 45 + 7 * S6 / 360, - 2 / 225 - S6 / 75],
            [4 / 9 - S6 / 36, 4 / 9 + S6 / 36, 1 / 9],
        ])
    else:
        raise NotImplementedError
    
    c = np.sum(A, axis=1)
    s = len(c) # stages
    n = len(y) # problem dimension

    # vander = np.vander([0, *c], increasing=True)[1:, 1:]
    # vander_inv = np.linalg.inv(vander)

    Z = None
    i = 0
    k_sum = 0
    while t <= t1:
        i += 1
        print(f"t: {t}")

        def residual(Z):
            Z = Z.reshape(s, -1)

            F = np.empty_like(Z)
            for i in range(s):
                F[i] = f(t + c[i] * h, y + Z[i])

            res = Z - h * A @ F
            return res.reshape(-1)
        
        def jac(Z):
            return approx_derivative(residual, Z, method="2-point")
        
        # if Z is not None:
        #     # TODO: This seems to be bad or wrong
        #     # extrapolation of the collocation polynomial
        #     p = vander_inv @ Z.reshape(2, -1)
        #     z1 = p[0] * c[0] + p[1] * c[0]**2
        #     z2 = p[0] * c[1] + p[1] * c[1]**2
        #     Z = np.concatenate((z1, z2))
        # else:
        #     # trivial initial guess
        #     Z = np.zeros(2 * n)

        # trivial initial guess
        Z = np.zeros(s * n)

        res = residual(Z)
        error = np.linalg.norm(res)
        # error = 1
        k = 0
        while error >= eps:
            k += 1
            Z -= np.linalg.solve(jac(Z), res)
            res = residual(Z)
            error = np.linalg.norm(res)
            # print(f"  * k: {k}")
            # print(f"  * error: {error}")

        y += Z.reshape(s, -1)[-1].copy()
        t += h
        k_sum += k
        # print(f" - newton iter: {k}")

        t_sol.append(t)
        y_sol.append(y.copy())

    print(f"newton iter average: {k_sum / i}")

    return OptimizeResult(
        t=np.array(t_sol), 
        y=np.array(y_sol).T,
    )


def SymplecticEuler(f, y0, t_span, h):
    t_sol = []
    y_sol = []
    y = y0.copy()
    t0, t1 = t_span
    t = t0
    while t <= t1:
        t += h
        y[:3] += h * f(t, y)[:3]
        y[3:] += h * f(t, y)[3:]
        print(f"t: {t}")

        t_sol.append(t)
        y_sol.append(y.copy())

    return OptimizeResult(
        t=np.array(t_sol), 
        y=np.array(y_sol).T,
    )


def Moreau(f, y0, t_span, h):
    t_sol = []
    y_sol = []
    y = y0.copy()
    t0, t1 = t_span
    t = t0
    while t <= t1:
        t += h
        y[:3] += 0.5 * h * f(t, y)[:3]
        y[3:] += h * f(t, y)[3:]
        y[:3] += 0.5 * h * f(t, y)[:3]
        print(f"t: {t}")

        t_sol.append(t)
        y_sol.append(y.copy())

    return OptimizeResult(
        t=np.array(t_sol), 
        y=np.array(y_sol).T,
    )


def ax2skew(a: np.ndarray) -> np.ndarray:
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)


def T_SO3(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    if angle > EPS:
        # Park2005 (19), actually its the transposed!
        sa = np.sin(angle)
        ca = np.cos(angle)
        psi_tilde = ax2skew(psi)
        alpha = sa / angle
        beta2 = (1.0 - ca) / angle2
        return (
            np.eye(3, dtype=float)
            - beta2 * psi_tilde
            + ((1.0 - alpha) / angle2) * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) - 0.5 * ax2skew(psi)


def T_SO3_inv(psi: np.ndarray) -> np.ndarray:
    angle2 = psi @ psi
    angle = np.sqrt(angle2)
    psi_tilde = ax2skew(psi)
    if angle > EPS:
        # Park2005 (19), actually its the transposed!
        gamma = 0.5 * angle / (np.tan(0.5 * angle))
        return (
            np.eye(3, dtype=float)
            + 0.5 * psi_tilde
            + ((1.0 - gamma) / angle2) * psi_tilde @ psi_tilde
        )
    else:
        # first order approximation
        return np.eye(3, dtype=float) + 0.5 * psi_tilde


def smoothstep5(x, x_min=0, x_max=1):
    """5th-order smoothstep function, see https://en.wikipedia.org/wiki/Smoothstep."""
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def cot(phi):
    """Cotangens, see https://de.wikipedia.org/wiki/Tangens_und_Kotangens."""
    return np.cos(phi) / np.sin(phi)


class SE3Rod:
    def __init__(self, scale_units=False):
        # geometry of the rod
        length = np.pi / 2
        # width = 1e-3 # [m]
        width = 1e-2 # [m]
        density = 8.e3  # [kg / m^3]

        # material properties
        if STEEL:
            E = 260.0e9 # [N / m^2]
            G = 100.0e9 # [N / m^2]
        else:
            E = 0.5 * (0.1 - 0.01) * 1e9
            G = 0.0006 * 1e9

        # time span
        t0 = 0
        t1 = 10

        self.scale_units = scale_units
        if scale_units:
            # [m] -> [mm]
            length *= 1e3
            # cross section properties
            width *= 1e3
            density *= 1e-9

            # [s] -> [ms]
            t0 *= 1e3
            t1 *= 1e3

            # [N / m^2] -> [kN / mm^2]
            E *= 1e-9
            G *= 1e-9

        self.E = E
        self.G = G

        self.length = length
        self.width = width
        self.density = density

        A = width**2
        I = width**4 / 12

        self.A = A
        self.I = I

        self.EA = E * A
        self.GA = G * A
        self.EI = E * I

        self.K = np.diag([self.EA, self.GA, self.EI])

        self.M = np.diag([A, A, I]) * density * length / 3
        self.M_inv = np.diag([1 / A, 1 / A, 1 / I]) / density / length * 3

        # initial conditions
        self.q0 = np.array([length, 0, 0], dtype=float)
        c = 0.125
        phi0 = c * 2 * np.pi
        self.q0 = np.array([length / phi0 * np.sin(phi0), length / phi0 * (1 - np.cos(phi0)), phi0], dtype=float)
        self.u0 = np.array([0, 0, 0], dtype=float)
        q_dot0 = self.u0
        u_dot0 = np.array([0, 0, 0], dtype=float)

        # reference strain measure
        self.epsilon0 = np.array([1, 0, 0], dtype=float)

        self.t0 = t0
        self.t1 = t1
        self.y0 = np.concatenate([self.q0, self.u0])
        self.yp0 = np.concatenate([q_dot0, u_dot0])

    def A_IK(self, xi, q):
        x, y, phi = q
        sphi, cphi = np.sin(xi * phi), np.cos(xi * phi)
        return np.array([
            [cphi, -sphi, 0],
            [sphi,  cphi, 0],
            [   0,     0, 1],
        ])

    def r_OP(self, xi, q):
        x, y, phi = q
        psi01 = np.array([0, 0, phi])
        psi = xi * psi01
        return (xi * T_SO3(psi).T @ T_SO3_inv(psi01).T @ np.array([x, y, 0]))[:2]

    def epsilon_bar(self, q):
        """Strain measures."""
        x, y, phi = q

        abs_phi = abs(phi)
        if abs_phi > EPS:
            g = 0.5 * abs_phi * cot(abs_phi / 2)
        else:
            # https://www.wolframalpha.com/input?i=taylor+series+abs%28x%29+cot%28abs%28x%29+%2F+2%29+%2F+2
            g = 1 - phi**2 / 12 - phi**4 / 720

        return np.array([
            x * g + 0.5 * phi * y,
            y * g - 0.5 * phi * x,
            phi,
        ])
    
    def W(self, q):
        x, y, phi = q

        if abs(phi) > EPS:
            sphi, cphi = np.sin(phi), np.cos(phi)
            diag = sphi / phi
            off_diag = (1 - cphi) / phi
        else:
            # https://www.wolframalpha.com/input?i=taylor+series+sin%28c+*+x%29+%2F+x
            diag = 1 - phi**2 / 6 + phi**4 / 120

            # https://www.wolframalpha.com/input?i=taylor+series+%281+-+cos%28c+*+x%29%29+%2F+x
            off_diag = phi / 2 - phi**3 / 24 + phi**5 / 720

        ga_x_bar, ga_y_bar, kappa_bar = self.epsilon_bar(q)
        
        return np.array([
            [          diag,       -off_diag, 0],
            [      off_diag,            diag, 0],
            [0.5 * ga_y_bar, -0.5 * ga_x_bar, 1],
        ])
        
    def la(self, q):
        return self.K @ (self.epsilon_bar(q) / self.length - self.epsilon0)

    def f_ext(self, t):
        m_max = 2 * np.pi * self.EI / self.length * 0.49
        # m_max = np.pi * self.EI / self.length * 0.2
        # # m = m_max * (
        # #     smoothstep5(t, self.t0, 0.5 * self.t1)
        # #     - smoothstep5(t, 0.5 * self.t1, 0.51 * self.t1)
        # # )

        if t <= 0.5 * t1:
            m = m_max * smoothstep5(t, self.t0, 0.5 * self.t1)
            # m = m_max * t / (0.5 * t1)
        else:
            m = 0

        m = 0

        # m = m_max * smoothstep5(t, self.t0, self.t1)
    
        return np.array([0, 0, m])

    def F(self, t, y, yp):
        q, u = y[:3], y[3:]
        qp, up = yp[:3], yp[3:]

        return np.concatenate([
            qp - u,
            self.M @ up + self.W(q) @ self.la(q) - self.f_ext(t),
        ])

    def f(self, t, y):
        q, u = y[:3], y[3:]

        return np.concatenate([
            u,
            self.M_inv @ (self.f_ext(t) - self.W(q) @ self.la(q)),
        ])
    
    def J(self, t, y):
        return approx_derivative(lambda y: self.f(t, y), y) 
        A = 0
        return np.block([
            [np.zeros((3, 3)),   np.eye((3, 3))],
            [               A, np.zeros((3, 3))],
        ])


if __name__ == "__main__":
    # rod finite element
    scale_units = False
    # scale_units = True
    rod = SE3Rod(scale_units)
    y0 = rod.y0
    yp0 = rod.yp0
    t0 = rod.t0
    t1 = rod.t1
    length = rod.length

    F0 = rod.F(t0, y0, yp0)
    print(f"F0: {F0}")

    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    y0, yp0, F0 = consistent_initial_conditions(rod.F, t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    # J0_num = rod.J(t0, y0)

    # np.set_printoptions(3, suppress=True)

    # EA, GA, EI, L = rod.EA, rod.GA, rod.EI, rod.length
    # fint_q = np.array([
    #     [EA / L, 0, 0],
    #     [0, GA / L, -GA / 2],
    #     [0, -GA / 2, GA * L / 4 + EI / L]
    # ])
    # J0_10 = -rod.M_inv @ fint_q
    # # print(f"J0_10:\n{J0_10}")

    # J0 = np.block([
    #     [   np.zeros((3, 3)),        np.eye(3)],
    #     [-rod.M_inv @ fint_q, np.zeros((3, 3))],
    # ])

    # print(f"J0_num:\n{J0_num}")
    # print(f"J0:\n{J0}")

    # assert np.allclose(J0, J0_num)

    # la, V = eig(J0)
    # print(f"la: {la}")

    # a, b, c = la[0].imag, la[2].imag, la[4].imag
    # print(f"a: {a:.3e}; b: {b:.3e}; c: {c:.3e}")

    # print(f"mu_12: {np.sqrt(3 * rod.E / (rod.length**2 * rod.density))}")

    # print(f"???: {-3 * (rod.A * rod.G * rod.length + 2 * rod.E * rod.I + 2 * rod.G * rod.I) / (4 * rod.I + rod.length**2 * rod.density)}")

    # cond = np.linalg.cond(J0)
    # print(f"cond: {cond:.3e}")

    # print(f"h_max_a: {2 / a:.3e}")
    # print(f"h_max_b: {2 / b:.3e}")
    # print(f"h_max_c: {2 / c:.3e}")

    # exit()

    ##############
    # solver setup
    ##############
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    t_eval = None

    # method = "BDF"
    method = "Radau"

    # atol = rtol = 1e-5
    atol = rtol = 1e-3

    ##############
    # dae solution
    ##############
    start = time.time()
    # sol = solve_dae(rod.F, t_span, y0, yp0, atol=atol, rtol=rtol, method=method, t_eval=t_eval, stages=3)
    # # sol = solve_ivp(rod.f, t_span, y0, method=method, t_eval=t_eval, atol=atol, rtol=rtol)
    # sol = EulerForward(rod.f, rod.y0, (0, 1e-3), h=1e-8)
    # sol = SymplecticEuler(rod.f, rod.y0, (0, 0.1), h=1.25e-6)
    # sol = Moreau(rod.f, rod.y0, (0, 0.1), h=1.25e-6)
    # sol = EulerBackward(rod.f, rod.y0, (0, 10), h=1e-3) # no step-size restrictions, but accuracy problems
    sol = RadauIIA(rod.f, rod.y0, (0, 10), h=1e-2, s=2) # no step-size restrictions, no visible damping for s=2 and h=1e-2
    # sol = RadauIIA(rod.f, rod.y0, (0, 10), h=5e-2, s=3) # no step-size restrictions, no visible damping for s=2 and h=1e-2
    end = time.time()
    t = sol.t
    y = sol.y
    # success = sol.success
    # status = sol.status
    # message = sol.message
    # print(f"message: {message}")
    # print(f"elapsed time: {end - start}")
    # print(f"nfev: {sol.nfev}")
    # print(f"njev: {sol.njev}")
    # print(f"nlu: {sol.nlu}")

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(5, 1)

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
    ax[3].set_xlabel("m(t)")
    ax[3].grid()
    ax[3].plot(t, np.array([rod.f_ext(ti)[-1] for ti in t]), "-ok")

    ax[4].set_xlabel("t")
    ax[4].set_xlabel("h")
    ax[4].grid()
    ax[4].plot(t[1:], np.diff(t), "-ok")
    ax[4].set_yscale("log")

    ###########
    # animation
    ###########

    # prepare data for animation
    frames = len(t)
    target_frames = min(frames, 100)
    # target_frames = frames
    frac = max(1, int(np.ceil(frames / target_frames)))
    animation_time = t1 / 2
    animation_time = 5
    interval = animation_time * 1000 / target_frames

    t = t[::frac]
    y = y[:, ::frac]

    target_frames = y.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5 * length, 1.5 * length)
    ax.set_ylim(-length, length)
    # ax.set_aspect('equal', 'box')
    ax.set_aspect('equal')

    centerline, = ax.plot([], [], "-k")
    node, = ax.plot([], [], "or")
    rect = Rectangle((-0.1 * length, -0.25 * length), 0.1 * length, 0.5 * length, hatch='//', edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    xis = np.linspace(0, 1, num=100)

    def update(t, y):
        q, u = y[:3], y[3:]
        r_OP = np.array([rod.r_OP(xi, q) for xi in xis])
        node.set_data([0, q[0]], [0, q[1]])
        centerline.set_data(*r_OP.T)
        return centerline,

    def animate(i):
        update(t[i], y[:, i])

    anim = FuncAnimation(
        fig, animate, frames=target_frames, interval=interval, blit=False, repeat=True
    )

    plt.show()
