import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import RigidBody, Frame, Cylinder, Box
from cardillo.math import axis_angle2quat, e1, e2, e3, cross3, norm, A_IB_basic
from cardillo.math.approx_fprime import approx_fprime
from cardillo.forces import Force


class RollingCondition:
    """Rolling condition for rigid disc:
    - impenetrability on position level.
    - nonholonomic no sliding on velocity level."""

    def __init__(self, subsystem, la_g0=None, la_gamma0=None):
        self.subsystem = subsystem

        self.nla_g = 1
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.nla_gamma = 2
        self.la_gamma0 = np.zeros(self.nla_gamma) if la_gamma0 is None else la_gamma0

    def assembler_callback(self):
        self.qDOF = self.subsystem.qDOF[self.subsystem.local_qDOF_P()]
        self.uDOF = self.subsystem.qDOF[self.subsystem.local_uDOF_P()]

    def r_CP(self, t, q):
        # evaluate body-fixed basis
        e_B_x, e_B_y, e_B_z = self.subsystem.A_IB(t, q).T

        # compute e_x axis of grinding-(G)-frame, see LeSaux2005 (2.11)
        g_G_x = cross3(e_B_y, e3)
        e_G_x = g_G_x / norm(g_G_x)

        # compute e_z axis of G-frame, see LeSaux2005 (2.12)
        e_G_z = cross3(e_G_x, e_B_y)

        # contact point is - radius * e_z axis of grinding frame, see LeSaux2005 (2.13)
        return -self.subsystem.radius * e_G_z

    #################
    # non penetration
    #################
    def g(self, t, q):
        # see LeSaux2005 (2.15a)
        r_OC = self.subsystem.r_OP(t, q)
        r_OC = r_OC + self.r_CP(t, q)
        return r_OC @ e3

    def g_dot(self, t, q, u):
        v_C = self.subsystem.v_P(
            t, q, u, B_r_CP=self.subsystem.A_IB(t, q).T @ self.r_CP(t, q)
        )
        return v_C @ e3

    def g_dot_u(self, t, q):
        return self.W_g(t, q).T

    def g_ddot(self, t, q, u, u_dot):
        g_dot_q = approx_fprime(
            q, lambda q: self.g_dot(t, q, u), method="cs", eps=1.0e-15
        )

        return g_dot_q @ self.subsystem.q_dot(t, q, u) + self.g_dot_u(t, q) @ u_dot

    def g_q(self, t, q):
        return approx_fprime(
            q, lambda q: self.g(t, q), method="cs", eps=1.0e-15
        ).reshape(self.nla_g, self.subsystem.nq)

    def g_qq_dense(self, t, q):
        return approx_fprime(q, lambda q: self.g_q(t, q), method="3-point").reshape(
            self.nla_g, self.subsystem.nq, self.subsystem.nq
        )

    def g_q_T_mu_q(self, t, q, mu_g):
        return np.einsum("ijk,i", self.g_qq_dense(t, q), mu_g)

    def g_dot_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.g_dot(t, q, u)).reshape(
            self.nla_g, self.subsystem.nq
        )

    def W_g(self, t, q):
        J_C = self.subsystem.J_P(
            t, q, B_r_CP=self.subsystem.A_IB(t, q).T @ self.r_CP(t, q)
        )
        return (e3 @ J_C).reshape(-1, self.nla_g)

    def Wla_g_q(self, t, q, la_g):
        return approx_fprime(q, lambda q: self.W_g(t, q) @ la_g)

    ########################
    # no in plane velocities
    ########################

    def gamma(self, t, q, u):
        v_C = self.subsystem.v_P(
            t, q, u, B_r_CP=self.subsystem.A_IB(t, q).T @ self.r_CP(t, q)
        )
        return np.array([v_C @ e1, v_C @ e2])

    def gamma_dot(self, t, q, u, u_dot):
        gamma_q = approx_fprime(
            q, lambda q: self.gamma(t, q, u), method="cs", eps=1.0e-15
        )
        gamma_u = self.gamma_u(t, q)

        return gamma_q @ self.subsystem.q_dot(t, q, u) + gamma_u @ u_dot

    def gamma_q(self, t, q, u):
        return approx_fprime(q, lambda q: self.gamma(t, q, u))

    def gamma_dot_q(self, t, q, u, u_dot):
        raise NotImplementedError("")

    def gamma_u(self, t, q):
        return self.subsystem.J_P(
            t, q, B_r_CP=self.subsystem.A_IB(t, q).T @ self.r_CP(t, q)
        )[:2]

    def W_gamma(self, t, q):
        return self.gamma_u(t, q).T

    def Wla_gamma_q(self, t, q, la_gamma):
        return approx_fprime(q, lambda q: self.gamma_u(t, q).T @ la_gamma)


def disc(mass, radius, q0=None, u0=None):
    width = radius / 100
    A = 1 / 4 * mass * radius**2
    C = 1 / 2 * mass * radius**2
    B_Theta_C = np.diag(np.array([A, C, A]))

    disc = Cylinder(RigidBody)(
        radius,
        height=width,
        A_BM=A_IB_basic(-np.pi / 2).x,
        B_r_CP=np.array([0, width / 2, 0]),
        mass=mass,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
    )
    return disc


def disc_boundary(disc, t, q, n=100):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
    B_r_CP = disc.radius * np.vstack([np.sin(phi), np.zeros(n), np.cos(phi)])
    return np.repeat(disc.r_OP(t, q), n).reshape(3, n) + disc.A_IB(t, q) @ B_r_CP


if __name__ == "__main__":
    """Analytical analysis of the roolling motion of a disc, see Lesaux2005
    Section 5 and 6.

    References
    ==========
    Lesaux2005: https://doi.org/10.1007/s00332-004-0655-4
    """

    ############
    # parameters
    ############
    gravity = 9.81  # gravity
    m = 0.3048  # disc mass

    # disc radius
    r = 0.05

    # radius of of the circular motion
    R = 10 * r  # used for GAMM

    # inertia of the disc, Lesaux2005 before (5.3)
    A = B = 0.25 * m * r**2
    C = 0.5 * m * r**2

    # ratio between disc radius and radius of rolling
    rho = r / R  # Lesaux2005 (5.10)

    ####################
    # initial conditions
    ####################
    beta0 = 5 * np.pi / 180  # initial inlination angle (0 < beta0 < pi/2)

    # center of mass
    x0 = 0
    y0 = R - r * np.sin(beta0)
    z0 = r * np.cos(beta0)

    # initial angles
    beta_dot0 = 0  # Lesaux1005 before (5.10)
    gamma_dot0_pow2 = (
        4
        * (gravity / r)
        * np.sin(beta0)
        / ((6 - 5 * rho * np.sin(beta0)) * rho * np.cos(beta0))
    )
    gamma_dot0 = np.sqrt(gamma_dot0_pow2)  # Lesaux2005 (5.12)
    alpha_dot0 = -rho * gamma_dot0  # Lesaux2005 (5.11)

    # angular velocity
    B_Omega0 = np.array(
        [beta_dot0, alpha_dot0 * np.sin(beta0) + gamma_dot0, alpha_dot0 * np.cos(beta0)]
    )

    # center of mass velocity
    v_C0 = np.array([-R * alpha_dot0 + r * alpha_dot0 * np.sin(beta0), 0, 0])

    # initial conditions
    t0 = 0
    p0 = axis_angle2quat(np.array([1, 0, 0]), beta0)
    q0 = np.array((x0, y0, z0, *p0))
    u0 = np.concatenate((v_C0, B_Omega0))

    #################
    # assemble system
    #################

    # create disc
    disc = disc(m, r, q0, u0)

    # create rolling condition
    rolling_condition = RollingCondition(disc)

    # gravity
    f_g = Force(lambda t: np.array([0, 0, -m * gravity]), disc)

    # create floor (Box only for visualization purposes)
    floor = Box(Frame)(
        dimensions=[2.2 * R, 2.2 * R, 0.0001],
        name="floor",
    )
    # assemble system
    system = System()
    system.add(disc, rolling_condition, f_g, floor)
    system.assemble()

    ############
    # simulation
    ############
    t1 = 2 * np.pi / np.abs(alpha_dot0)  # simulation time
    dt = 2.0e-2  # time step

    # sol = ScipyIVP(system, t1, dt).solve()
    # sol = Rattle(system, t1, dt).solve()
    sol = ScipyDAE(system, t1, dt).solve()

    # read solution
    t = sol.t  # time
    q = sol.q  # position coordinates
    u = sol.u  # velocity coordinates

    # compute bilateral constraint quantities
    g = np.array([system.g(ti, qi) for ti, qi in zip(t, q)])
    g_dot = np.array([system.g_dot(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])
    gamma = np.array([system.gamma(ti, qi, ui) for ti, qi, ui in zip(t, q, u)])

    #################
    # post-processing
    #################

    # plots
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 7))
    fig.suptitle("Evolution of constraint quantities")
    # g
    ax[0].plot(t, g)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("g")
    ax[0].grid()

    # g_dot
    ax[1].plot(t, g_dot)
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("$\dot{g}$")
    ax[1].grid()

    # gamma_x
    ax[2].plot(t, gamma[:, 0])
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("$\gamma_x$")
    ax[2].grid()

    # gamma_y
    ax[3].plot(t, gamma[:, 1])
    ax[3].set_xlabel("t")
    ax[3].set_ylabel("$\gamma_y$")
    ax[3].grid()

    plt.tight_layout()
    plt.show()

    # animation
    t = t
    q = q

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    scale = R
    ax.set_xlim3d(left=-scale, right=scale)
    ax.set_ylim3d(bottom=-scale, top=scale)
    ax.set_zlim3d(bottom=0, top=2 * scale)

    from collections import deque

    slowmotion = 1
    fps = 200
    animation_time = slowmotion * t1
    target_frames = int(fps * animation_time)
    frac = max(1, int(len(t) / target_frames))
    if frac == 1:
        target_frames = len(t)
    interval = 1000 / fps

    frames = target_frames
    t = t[::frac]
    q = q[::frac]

    def create(t, q):
        x_S, y_S, z_S = disc.r_OP(t, q)

        A_IB = disc.A_IB(t, q)
        d1 = A_IB[:, 0] * r
        d2 = A_IB[:, 1] * r
        d3 = A_IB[:, 2] * r

        (COM,) = ax.plot([x_S], [y_S], [z_S], "ok")
        (bdry,) = ax.plot([], [], [], "-k")
        (trace,) = ax.plot([], [], [], "--k")
        (d1_,) = ax.plot(
            [x_S, x_S + d1[0]], [y_S, y_S + d1[1]], [z_S, z_S + d1[2]], "-r"
        )
        (d2_,) = ax.plot(
            [x_S, x_S + d2[0]], [y_S, y_S + d2[1]], [z_S, z_S + d2[2]], "-g"
        )
        (d3_,) = ax.plot(
            [x_S, x_S + d3[0]], [y_S, y_S + d3[1]], [z_S, z_S + d3[2]], "-b"
        )

        return COM, bdry, trace, d1_, d2_, d3_

    COM, bdry, trace, d1_, d2_, d3_ = create(0, q[0])

    def update(t, q, COM, bdry, trace, d1_, d2_, d3_):
        global x_trace, y_trace, z_trace
        if t == t0:
            x_trace = deque([])
            y_trace = deque([])
            z_trace = deque([])

        x_S, y_S, z_S = disc.r_OP(t, q)

        x_bdry, y_bdry, z_bdry = disc_boundary(disc, t, q)

        x_t, y_t, z_t = disc.r_OP(t, q) + rolling_condition.r_CP(t, q)

        x_trace.append(x_t)
        y_trace.append(y_t)
        z_trace.append(z_t)

        A_IB = disc.A_IB(t, q)
        d1 = A_IB[:, 0] * r
        d2 = A_IB[:, 1] * r
        d3 = A_IB[:, 2] * r

        COM.set_data(np.array([x_S]), np.array([y_S]))
        COM.set_3d_properties(np.array([z_S]))

        bdry.set_data(np.array(x_bdry), np.array(y_bdry))
        bdry.set_3d_properties(np.array(z_bdry))

        trace.set_data(np.array(x_trace), np.array(y_trace))
        trace.set_3d_properties(np.array(z_trace))

        d1_.set_data(np.array([x_S, x_S + d1[0]]), np.array([y_S, y_S + d1[1]]))
        d1_.set_3d_properties(np.array([z_S, z_S + d1[2]]))

        d2_.set_data(np.array([x_S, x_S + d2[0]]), np.array([y_S, y_S + d2[1]]))
        d2_.set_3d_properties(np.array([z_S, z_S + d2[2]]))

        d3_.set_data(np.array([x_S, x_S + d3[0]]), np.array([y_S, y_S + d3[1]]))
        d3_.set_3d_properties(np.array([z_S, z_S + d3[2]]))

        return COM, bdry, trace, d1_, d2_, d3_

    def animate(i):
        update(t[i], q[i], COM, bdry, trace, d1_, d2_, d3_)

    anim = animation.FuncAnimation(
        fig, animate, frames=frames, interval=interval, blit=False
    )

    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    e = system.export(dir_name, "vtk", sol)
    # additionally export body fixed frame
    e.export_contr(disc, file_name="A_IB", base_export=True)
