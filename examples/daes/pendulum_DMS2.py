import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
from scipy._lib._util import _RichResult
import matplotlib.pyplot as plt


def solve_mechanical_system(system, q0, u0, t_span, atol=1e-6, rtol=1e-6):
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    
    nq = len(q0)
    nu = len(u0)
    W0 = system.W(t0, q0)
    nla = W0.shape[1]

    O = np.zeros((nla, nla))

    y0 = np.concatenate([q0, u0])

    frac = (t1 - t0) / 101
    pbar = tqdm(total=100, leave=True)
    global i
    i = 0
    
    def eqm(t, y):
        global i
        i1 = int(t // frac)
        pbar.update(i1 - i)
        pbar.set_description(f"t: {t:0.2e}s < {t1:0.2e}s")
        i = i1

        q = y[:nq]
        u = y[nq:]

        M = system.M(t, q)
        h = system.h(t, q, u)
        W = system.W(t, q)
        zeta = system.zeta(t, q, u)

        A = np.bmat([
            [  M, -W],
            [W.T,  O],
        ])
        b = np.concatenate([
            h,
            -zeta,
        ])

        q_dot = u
        u_dot, la = np.split(np.linalg.solve(A, b), [nu])
        return np.concatenate([q_dot, u_dot])
    
    sol = solve_ivp(
        eqm,
        t_span,
        y0,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    t = sol.t
    q = sol.y[:nq, :]
    u = sol.y[nq :, :]

    return _RichResult(
        t=t,
        q=q,
        u=u,
        sol=sol,
    )


m = 1
l = 1
g = 10

class CartesianPendulum:
    def __init__(self, m=1, l=1, g=10):
        self.m = m
        self.l = l
        self.g = g

    def M(self, t, q):
        return np.diag([self.m, self.m])
    
    def h(self, t, q, u):
        return np.array([0, -self.m * self.g])
    
    def zeta(self, t, q, u):
        ux, uy = u
        return np.array([2 * ux**2 + 2 * uy**2])
    
    def W(self, t, q):
        x, y = q
        return np.array([[2 * x], [2 * y]])


if __name__ == "__main__":
    # time span
    t0 = 0
    t = 10
    t_span = (t0, t)

    # mechanical system
    system = CartesianPendulum()

    # initial conditions
    q0 = np.array([1, 0])
    u0 = np.array([0, 0])

    ##############
    # dae solution
    ##############
    sol1 = solve_mechanical_system(system, q0, u0, t_span)
    t = sol1.t
    q = sol1.q
    u = sol1.u

    # visualization
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, q[0], "-", label="x")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, q[1], "-", label="y")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(t, q[0]**2 + q[1]**2 - l**2, "-", label="g")
    ax[2].legend()
    ax[2].grid()

    ax[3].plot(t, 2 * q[0] * u[0] + 2 * q[1] * u[1], "-", label="g_dot")
    ax[3].legend()
    ax[3].grid()

    plt.show()
