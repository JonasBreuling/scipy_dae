import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


def ax2skew(a):
    """Computes the skew symmetric matrix from a 3D tuple."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on


def quat2trafo(P):
    """Computes the transformation  matrix for a given quaternion."""
    p0, p = np.array_split(P, [1])
    p_tilde = ax2skew(p)
    P2 = P @ P
    return np.eye(3, dtype=P.dtype) + (2 / P2) * (p0 * p_tilde + p_tilde @ p_tilde)


def Spurrier(A):
    """
    Extract the unit quaternion from a given transformation matrix using 
    Spurrier's algorithm.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = np.zeros(4, dtype=float)
    decision[:3] = np.diag(A)
    decision[3] = np.trace(A)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=float)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * A[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (A[k, j] - A[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (A[j, i] + A[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (A[k, i] + A[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (A[2, 1] - A[1, 2]) / (4 * quat[0])
        quat[2] = (A[0, 2] - A[2, 0]) / (4 * quat[0])
        quat[3] = (A[1, 0] - A[0, 1]) / (4 * quat[0])

    return quat


class RigidBody:
    def __init__(self, mass, K_Omega_S, r_OS0, A_IK0, v_S0, K_omega_IK0, stabilize=None):
        self.mass = mass
        self.K_Omega_S = K_Omega_S
        
        self.q0 = RigidBody.pose2q(r_OS0, A_IK0)
        self.u0 = np.concatenate([v_S0, K_omega_IK0])

        if stabilize in [None, "project", "elastic"]:
            self.y0 = np.array([*self.q0, *self.u0])
        elif stabilize == "multiplier":
            self.y0 = np.array([*self.q0, *self.u0, 0])
        else:
            raise NotImplementedError
        
        if stabilize == "project":
            self.events = [self.__event]
        else:
            self.events = None

        self.stabilize = stabilize

    @staticmethod
    def pose2q(r_OS, A_IK):
        return np.concatenate([r_OS, Spurrier(A_IK)])
    
    def __call__(self, t, y, yp):
        if stabilize in [None, "project", "elastic"]:
            q, u = np.array_split(y, [7])
            q_dot, u_dot = np.array_split(yp, [7])
        elif stabilize == "multiplier":
            q, u, mu = np.array_split(y, [7, 13])
            q_dot, u_dot, mu_dot = np.array_split(yp, [7, 13])

        r_OS, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_S, B_omega_IK = np.array_split(u, [3])

        r_OS_dot, P_dot = np.array_split(q_dot, [3])
        a_S, K_psi_IK = np.array_split(u_dot, [3])

        # residual
        F = np.zeros_like(y)

        # kinematic differential equation
        F[:3] = r_OS_dot - v_S
        F[3:7] = P_dot - 0.5 * np.vstack((-p.T, p0 * np.eye(3, dtype=P.dtype) + ax2skew(p))) @ B_omega_IK

        # stabilize quaternion
        if self.stabilize == "elastic":
            # c = 0.1
            c = 1.5
            F[3:7] -= 2 * P * c * (1 - P @ P)
        if stabilize == "multiplier":
            F[3:7] -= 2 * P * mu_dot
            F[13] = P @ P - 1

        # equations of motion
        F[7:10] = self.mass * a_S # no external forces
        F[10:13] = self.K_Omega_S @ K_psi_IK + ax2skew(B_omega_IK) @ self.K_Omega_S @ B_omega_IK

        return F

    def __event(self, t, y):
        # project quaternion to be of unit length
        y[3:7] = y[3:7] / np.linalg.norm(y[3:7])

        return 1


if __name__ == "__main__":
    ############
    # parameters
    ############

    # simulation parameters
    t0 = 0  # initial time
    t1 = 10  # final time

    # inertia properties
    mass = 1
    A = 3
    B = 2.5
    C = 1
    K_Omega_S = np.diag([A, B, C])

    # initial conditions
    r_OS0 = np.zeros(3)
    A_IK0 = np.eye(3)
    v_S0 = np.zeros(3)
    epsilon = 1e-10
    omega_dot0 = 20
    K_omega_IK0 = np.array((epsilon, omega_dot0, 0))

    # create rigid body system
    # stabilize = None # no stabilization
    # stabilize = "elastic" # elastic stabilization
    # stabilize = "project" # projection
    stabilize = "multiplier" # Lagrange multiplier
    rigid_body = RigidBody(mass, K_Omega_S, r_OS0, A_IK0, v_S0, K_omega_IK0, stabilize=stabilize)

    # initial conditions
    y0 = rigid_body.y0
    yp0 = np.zeros_like(rigid_body.y0)
    y0, yp0, F0 = consistent_initial_conditions(rigid_body, t0, rigid_body.y0, np.zeros_like(rigid_body.y0))

    ##############
    # solver setup
    ##############
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    # t_eval = None

    method = "Radau"
    # method = "BDF"

    rtol = 1e-4
    atol = 1e-4

    ##############
    # dae solution
    ##############
    start = time.time()
    sol = solve_dae(rigid_body, t_span, y0, yp0, t_eval=t_eval, events=rigid_body.events, method=method, rtol=rtol, atol=atol)
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

    #################
    # post-processing
    #################
    q, u = np.array_split(y.T, [7], axis=1)
    r_OC = np.array([qi[:3] for qi in q])
    P = np.array([qi[3:] for qi in q])
    A_IK = np.array([quat2trafo(qi[3:]) for qi in q])
    e_yB = A_IK[:, :, 1]

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(3, 1, figsize=(10, 7))
    ax[0].set_title("Evolution of center of mass")
    ax[0].plot(t, r_OC[:, 0], label="x")
    ax[0].plot(t, r_OC[:, 1], label="y")
    ax[0].plot(t, r_OC[:, 2], label="z")
    ax[0].set_xlabel("t")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_title("Evolution of body-fixed y-axis")
    ax[1].plot(t, e_yB[:, 0], label="$(e_{y}^K)_x$")
    ax[1].plot(t, e_yB[:, 1], label="$(e_{y}^K)_y$")
    ax[1].plot(t, e_yB[:, 2], label="$(e_{y}^K)_z$")
    ax[1].set_xlabel("t")
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title("Quaternion length")
    ax[2].plot(t, np.linalg.norm(P, axis=1), label="$\|P\|$")
    ax[2].set_xlabel("t")
    ax[2].grid()
    ax[2].legend()

    plt.tight_layout()

    ###########
    # animation
    ###########
    def create_cuboid(I_r_OC, size, A_IK):
        # size of the cuboid along each axis
        dx, dy, dz = size
        
        # relative positions of the cuboid's corners
        K_r_CP = np.array([
            [-dx/2, -dy/2, -dz/2],
            [ dx/2, -dy/2, -dz/2],
            [ dx/2,  dy/2, -dz/2],
            [-dx/2,  dy/2, -dz/2],
            [-dx/2, -dy/2,  dz/2],
            [ dx/2, -dy/2,  dz/2],
            [ dx/2,  dy/2,  dz/2],
            [-dx/2,  dy/2,  dz/2],
        ])
        
        # compure corner points w.r.t. origin in inertial basis
        I_corners = I_r_OC + (A_IK @ K_r_CP.T).T
        
        return I_corners

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-4, 4)

    # initialize the elements to be updated
    origin_point, = ax.plot([], [], [], 'ko')
    x_axis_line, = ax.plot([], [], [], 'r-', label='e_x^K')
    y_axis_line, = ax.plot([], [], [], 'g-', label='e_y^K')
    z_axis_line, = ax.plot([], [], [], 'b-', label='e_z^K')
    cuboid_poly = Poly3DCollection([], alpha=0.3, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(cuboid_poly)

    def init():
        """Initialize the animation."""
        origin_point.set_data([], [])
        origin_point.set_3d_properties([])
        
        x_axis_line.set_data([], [])
        x_axis_line.set_3d_properties([])
        
        y_axis_line.set_data([], [])
        y_axis_line.set_3d_properties([])
        
        z_axis_line.set_data([], [])
        z_axis_line.set_3d_properties([])
        
        cuboid_poly.set_verts([])
        
        return origin_point, x_axis_line, y_axis_line, z_axis_line, cuboid_poly

    def update(i):
        """Update the elements for each frame."""
        r_OSi = r_OC[i]
        e_xi, e_yi, e_zi = A_IK[i].T
        A_IKi = A_IK[i]
        
        # update origin
        origin_point.set_data([r_OSi[0]], [r_OSi[1]])
        origin_point.set_3d_properties([r_OSi[2]])
        
        # update e_x^K
        x_end = r_OSi + e_xi
        x_axis_line.set_data([r_OSi[0], x_end[0]], [r_OSi[1], x_end[1]])
        x_axis_line.set_3d_properties([r_OSi[2], x_end[2]])
        
        # update e_y^K
        y_end = r_OSi + e_yi
        y_axis_line.set_data([r_OSi[0], y_end[0]], [r_OSi[1], y_end[1]])
        y_axis_line.set_3d_properties([r_OSi[2], y_end[2]])
        
        # update e_z^K
        z_end = r_OSi + e_zi
        z_axis_line.set_data([r_OSi[0], z_end[0]], [r_OSi[1], z_end[1]])
        z_axis_line.set_3d_properties([r_OSi[2], z_end[2]])
        
        # update cuboid
        M = mass / 12 * np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        b = np.array([A, B, C])
        size = np.sqrt(np.linalg.solve(M, b))

        cuboid_vertices = create_cuboid(r_OSi, size, A_IKi)
        cuboid_faces = [
            [cuboid_vertices[j] for j in [0, 1, 5, 4]],
            [cuboid_vertices[j] for j in [1, 2, 6, 5]],
            [cuboid_vertices[j] for j in [2, 3, 7, 6]],
            [cuboid_vertices[j] for j in [3, 0, 4, 7]],
            [cuboid_vertices[j] for j in [0, 1, 2, 3]],
            [cuboid_vertices[j] for j in [4, 5, 6, 7]],
        ]
        cuboid_poly.set_verts(cuboid_faces)
        
        return origin_point, x_axis_line, y_axis_line, z_axis_line, cuboid_poly

    # create the animation
    frames = len(t)
    interval = (t[-1] - t[0]) / frames * 1000
    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=interval, blit=False)

    plt.show()