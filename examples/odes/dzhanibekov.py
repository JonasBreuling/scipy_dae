import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy_dae.integrate import solve_dae, consistent_initial_conditions


def ax2skew(a):
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on


def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.199) and Nuetzi2016 (3.31).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165
    """
    p0, p = np.array_split(P, [1])
    p_tilde = ax2skew(p)
    if normalize:
        P2 = P @ P
        return np.eye(3, dtype=P.dtype) + (2 / P2) * (p0 * p_tilde + p_tilde @ p_tilde)
    else:
        return np.eye(3, dtype=P.dtype) + 2 * (p0 * p_tilde + p_tilde @ p_tilde)


def Spurrier(A):
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

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
    def __init__(self, mass, B_Omega_C, r_OC0, A_IB0, v_C0, B_omega_IB0, stabilize=None):
        self.mass = mass
        self.B_Omega_C = B_Omega_C
        
        self.q0 = RigidBody.pose2q(r_OC0, A_IB0)
        self.u0 = np.concatenate([v_C0, B_omega_IB0])

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
    def pose2q(r_OC, A_IB):
        return np.concatenate([r_OC, Spurrier(A_IB)])
    
    def __call__(self, t, y, yp):
        if stabilize in [None, "project", "elastic"]:
            q, u = np.array_split(y, [7])
            q_dot, u_dot = np.array_split(yp, [7])
        elif stabilize == "multiplier":
            q, u, mu = np.array_split(y, [7, 13])
            q_dot, u_dot, mu_dot = np.array_split(yp, [7, 13])

        r_OC, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_C, B_omega_IB = np.array_split(u, [3])

        r_OC_dot, P_dot = np.array_split(q_dot, [3])
        a_C, B_psi_IB = np.array_split(u_dot, [3])

        # residual
        F = np.zeros_like(y)

        # kinematic differential equation
        F[:3] = r_OC_dot - v_C
        F[3:7] = P_dot - 0.5 * np.vstack((-p.T, p0 * np.eye(3, dtype=P.dtype) + ax2skew(p))) @ B_omega_IB

        # stabilize quaternion
        if self.stabilize == "elastic":
            c = 1.5
            F[3:7] -= 2 * P * c * (1 - P @ P)
        if stabilize == "multiplier":
            F[3:7] -= 2 * P * mu_dot
            F[13] = P @ P - 1

        # equations of motion
        F[7:10] = self.mass * a_C
        F[10:13] = self.B_Omega_C @ B_psi_IB + ax2skew(B_omega_IB) @ self.B_Omega_C @ B_omega_IB

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
    B = 2
    C = 1
    B_Omega_C = np.diag([A, B, C])

    # initial conditions
    r_OC0 = np.zeros(3)
    A_IB0 = np.eye(3)
    v_C0 = np.zeros(3)
    epsilon = 1e-10
    omega_dot0 = 20
    B_omega_IB0 = np.array((epsilon, omega_dot0, 0))

    # create rigid body system
    stabilize = None # no stabilization
    # stabilize = "elastic" # elastic stabilization
    # stabilize = "multiplier" # Lagrange multiplier
    # stabilize = "project" # projection
    rigid_body = RigidBody(mass, B_Omega_C, r_OC0, A_IB0, v_C0, B_omega_IB0, stabilize=stabilize)

    # initial conditions
    y0 = rigid_body.y0
    yp0 = np.zeros_like(rigid_body.y0)
    F0 = rigid_body(t0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    y0, yp0, F0 = consistent_initial_conditions(rigid_body, t0, rigid_body.y0, np.zeros_like(rigid_body.y0))
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    ##############
    # solver setup
    ##############
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(1e3))
    # t_eval = None

    method = "Radau"
    # method = "BDF"

    rtol = 1e-3
    atol = 1e-3

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
    e_zB = np.array([Exp_SO3_quat(qi[3:])[:, 1] for qi in q])

    ########
    # export
    ########
    header = "t, x, y, z, e_y1, e_y2, e_y3, p0, p1, p2, p3, norm_P"

    data = np.concatenate((
        t[:, None],
        r_OC,
        e_zB,
        P,
        np.linalg.norm(P, axis=1)[:, None],
    ), axis=1)

    path = Path(sys.modules["__main__"].__file__)

    np.savetxt(
        path.parent / (path.stem + ".txt"),
        data,
        delimiter=", ",
        header=header,
        comments="",
    )

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
    ax[1].plot(t, e_zB[:, 0], label="$(e_{z}^B)_x$")
    ax[1].plot(t, e_zB[:, 1], label="$(e_{z}^B)_y$")
    ax[1].plot(t, e_zB[:, 2], label="$(e_{z}^B)_z$")
    ax[1].set_xlabel("t")
    ax[1].grid()
    ax[1].legend()

    ax[2].set_title("Quaternion length")
    ax[2].plot(t, np.linalg.norm(P, axis=1), label="$\|P\|$")
    ax[2].set_xlabel("t")
    ax[2].grid()
    ax[2].legend()

    plt.tight_layout()
    plt.show()
